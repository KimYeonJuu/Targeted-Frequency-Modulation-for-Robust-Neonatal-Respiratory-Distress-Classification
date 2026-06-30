import argparse
import glob
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from main_code.SpectralGatingNetwork import SpectralGatingNetwork, set_sgn_defaults
from sci.augmentation.lung_mixup import lung_mixup
from sci.augmentation.mixup import CrossEntropyLoss, guidedmix, mixup, stylemix
from sci.augmentation.outlier_band_aug import OutlierBandAug
from sci.dataset import CheXpertImageDataset, RDSImageDataset
from sci.inception_transformer import (
    iformer_base,
    iformer_base_384,
    iformer_small,
    iformer_small_384,
)
from main_code.medclip_binary import (
    MedCLIPBinaryClassifier,
    MedCLIPBinaryClassifier_SGN,
    MedCLIPBinaryClassifier_SGN_Inserted,
)
from sci.utils import (
    compute_accuracy,
    compute_auprc,
    compute_auroc,
    compute_f1,
    parse_hw,
)
from tqdm import tqdm


# Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------------------
# Test-only
# --------------------------------------------------------------------------------------
def test_only(
    args,
    model: nn.Module,
    device: torch.device,
    ckpt_dir: str,
    dataset: str,
    val_transform,
    batch_size: int = 32,
    use_wandb: bool = False,
) -> None:
    # Load the latest checkpoint.
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}/*.pth.")
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    print(f"[Test] Loaded checkpoint: {latest_ckpt}")

    # Test dataset
    if dataset == "chexpert":
        test_dataset = CheXpertImageDataset(
            split="test", data_path=args.test_data_path, transform=val_transform
        )
    elif dataset == "rds":
        test_dataset = RDSImageDataset(
            split="test", data_path=args.test_data_path, transform=val_transform
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=worker_init,
    )

    # Evaluation
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_labels, test_preds, test_probs = [], [], []

    with torch.no_grad():
        for images, labels, mask in tqdm(test_loader, desc="[Test]"):
            images = images.to(device)
            labels_idx = labels.argmax(dim=1).to(device)
            masks = mask.to(device)

            # Inject the external lung mask into SGN modules.
            lung_mask = masks
            if lung_mask.dim() == 3:
                lung_mask = lung_mask.unsqueeze(1)
            lung_mask = (lung_mask > 0).float()

            target_model = model.module if isinstance(model, nn.DataParallel) else model
            for m in target_model.modules():
                if isinstance(m, SpectralGatingNetwork):
                    m.set_external_mask(lung_mask)
            # ============================

            logits = model(images)
            loss = criterion(logits, labels_idx)
            test_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            test_labels.extend(labels_idx.cpu().tolist())
            test_preds.extend(preds.cpu().tolist())
            test_probs.extend(probs.cpu().tolist())

    avg_test_loss = test_loss / len(test_loader)
    test_acc = compute_accuracy(test_labels, test_preds)
    test_f1 = compute_f1(test_labels, test_preds)
    test_auroc = compute_auroc(test_labels, test_probs)
    test_auprc = compute_auprc(test_labels, test_probs)

    print(
        f"Test Loss: {avg_test_loss:.4f}, "
        f"Acc: {test_acc:.4f}, F1: {test_f1:.4f}, "
        f"AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}"
    )

    if use_wandb:
        wandb.log(
            {
                "test_loss": avg_test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_auroc": test_auroc,
                "test_auprc": test_auprc,
            }
        )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CNN Image-only Training & Testing")

    # Paths
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training CSV")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation CSV")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test CSV")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--save_model", type=str, default="model", help="Run name used for saved checkpoints")

    # Runtime settings
    parser.add_argument("--project_name", type=str, default="SCI-CheXpert", help="Weights & Biases project name")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_parallel", action="store_true", help="Enable torch.nn.DataParallel")
    parser.add_argument("--gpu", type=str, default="0", help='GPU IDs to use, e.g., "0" or "0,1"')

    # Model and data
    parser.add_argument("--model_name", type=str, required=True, help="Model name, e.g., tf_efficientnet_b0")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--freeze_backbone",
        type=str,
        default="OFF",
        choices=["ON", "OFF"],
        help='"ON" freezes the backbone, "OFF" enables full fine-tuning',
    )

    # SGN replacement/insertion specification
    parser.add_argument(
        "--block-selection",
        "--sb",
        dest="sb_spec",
        type=str,
        default="",
        help="Spectral block replacement specification using 1-based indices, e.g., '1:1,1:2;2:1' or '1:*;2:1-2'",
    )

    # SGN hyperparameters
    parser.add_argument("--sgn_mode", choices=["grid", "block"], default="grid", help="SGN frequency-processing mode")
    parser.add_argument("--sgn_grid_size", type=str, default="16x16", help="Fixed grid size for grid mode, formatted as HxW")
    parser.add_argument("--sgn_tau", type=float, default=0.25, help="Soft gating temperature")
    parser.add_argument("--sgn_amp", type=float, default=1.0, help="Modulation strength, scale = 1 + amp * gate")
    parser.add_argument("--sgn_block", type=str, default="16x16", help="Block size for block mode, formatted as HxW")
    parser.add_argument("--sgn_stride", type=str, default="same", help='Block-mode stride, formatted as HxW; "same" matches the block size')

    # SGN outlier-selection hyperparameters
    parser.add_argument("--sgn_ratio_thresh", type=float, default=0.02, help="Energy-ratio threshold")
    parser.add_argument("--sgn_outlier_topk", type=int, default=0, help="Select only the top-K frequencies; 0 disables top-K selection")
    parser.add_argument("--sgn_min_radius", type=int, default=1, help="Minimum radius for excluding low-frequency bins")
    parser.add_argument("--sgn_max_radius", type=int, default=-1, help="Maximum radius; -1 disables the upper bound")
    parser.add_argument("--sgn_mask_off", action="store_true", help="Disable outlier masking")

    # Optional modules and augmentations
    parser.add_argument("--use_cbam", action="store_true", help="Enable CBAM")
    parser.add_argument("--use_gtfe", action="store_true", help="Enable GTFE")

    parser.add_argument("--grid_shuffle_2_0_5", action="store_true", help="Enable Grid Shuffle")
    parser.add_argument("--grid_shuffle_3_0_5", action="store_true", help="Enable Grid Shuffle")
    parser.add_argument("--grid_shuffle_2_1_0", action="store_true", help="Enable Grid Shuffle")
    parser.add_argument("--grid_shuffle_3_1_0", action="store_true", help="Enable Grid Shuffle")

    parser.add_argument("--mixup", action="store_true", help="Enable MixUp")
    parser.add_argument("--stylemix", action="store_true", help="Enable StyleMix")
    parser.add_argument("--guidedmix", action="store_true", help="Enable GuidedMix")
    parser.add_argument("--lungmix", action="store_true", help="Enable LungMix")
    parser.add_argument("--mix_prob", type=float, default=0.5, help="Probability of applying mix-based augmentation")
    parser.add_argument("--alpha", type=float, default=1.0, help="Mixup alpha")
    parser.add_argument("--r", type=float, default=0.5, help="StyleMix r")
    parser.add_argument("--condition", type=str, default="greedy", help="GuidedMix condition")
    parser.add_argument("--saliency_mode", type=str, default="grad", help="GuidedMix saliency mode")

    parser.add_argument("--fft_aug", action="store_true", help="Enable 8x8 block FFT augmentation")
    parser.add_argument("--fft_aug_prob", type=float, default=0.5, help="Probability of applying FFT augmentation")
    parser.add_argument("--stride_1", action="store_true", help="Use stride 1")

    # Outlier-Band Aug
    parser.add_argument("--outlier_csv", type=str, default="", help="Path to the outlier-band CSV")
    parser.add_argument("--outlier_prob", type=float, default=0.5)
    parser.add_argument("--outlier_scale", type=float, default=2.0)
    parser.add_argument("--outlier_topk", type=int, default=0)
    parser.add_argument("--outlier_subset_k", type=int, default=0)
    parser.add_argument("--inner_margin", type=int, default=1)
    parser.add_argument("--pre_apodize", type=float, default=1.0)
    parser.add_argument("--post_hard_mask", action="store_true", help="Keep the background zero after IFFT")

    args = parser.parse_args()

    # Set SGN defaults.
    bh, bw = parse_hw(args.sgn_block) or (16, 16)
    stride = parse_hw(args.sgn_stride)
    gh, gw = parse_hw(args.sgn_grid_size) or (16, 16)

    set_sgn_defaults(
        mode=args.sgn_mode,
        tau=args.sgn_tau,
        amp=args.sgn_amp,
        block_size=(bh, bw),
        block_stride=stride,
        grid_size=(gh, gw),
        use_outlier_mask=not args.sgn_mask_off,
        ratio_thresh=args.sgn_ratio_thresh,
        outlier_topk=args.sgn_outlier_topk,
        min_radius=args.sgn_min_radius,
        max_radius=args.sgn_max_radius,
    )
    # ========================

    # GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")

    # Default action: train.
    if not args.train and not args.test:
        args.train = True

    set_seed(args.seed)

    # Checkpoint path
    ckpt_root = args.checkpoint_dir
    ckpt_dir = os.path.join(ckpt_root, args.model_name, args.save_model)
    print(f"ckpt_dir : {ckpt_dir}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Device and freeze settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_flag = args.freeze_backbone.upper() == "ON"

    # Model construction
    if args.model_name == "iformer_small":
        model = iformer_small(pretrained=True, num_classes=2, use_cbam=args.use_cbam, use_gtfe=args.use_gtfe)
        if args.stride_1:
            model.patch_embed.proj1.stride = (1, 1)

    elif args.model_name == "medclip_vit":
        model = MedCLIPBinaryClassifier(backbone="vit", num_classes=2)
        if freeze_flag:
            for p in model.backbone.parameters():
                p.requires_grad = False

    elif args.model_name == "medclip_resnet":
        model = MedCLIPBinaryClassifier(backbone="resnet", num_classes=2)
        if freeze_flag:
            for p in model.backbone.parameters():
                p.requires_grad = False

    elif args.model_name == "medclip_swin_sgn":
        model = MedCLIPBinaryClassifier_SGN(
            num_classes=2, 
            sb_spec=args.sb_spec, 
            freeze_backbone=freeze_flag
            )

    elif args.model_name == "medclip_swin_sgn_insert":
        model = MedCLIPBinaryClassifier_SGN_Inserted(
            num_classes=2, 
            sb_insert_spec=args.sb_spec, 
            freeze_backbone=freeze_flag
        )

    else:
        model = timm.create_model(args.model_name, pretrained=True, num_classes=2)

    print(model)

    # DataParallel
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"[Info] DataParallel enabled, GPU: {args.gpu}")
        model = nn.DataParallel(model)

    model.to(device)

    # Initialize Weights & Biases.
    run = None
    if args.use_wandb and (args.train or args.test):
        run = wandb.init(
        project=args.project_name,
        name=args.save_model,
        config=vars(args),
        reinit=True,
        save_code=False,   # Code artifacts are added explicitly below.
    )
        
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing main.py
    except NameError:
        script_dir = os.getcwd()

    main_code_dir = os.path.join(script_dir, "main_code")

    # Add the full folder.
    if run is not None and os.path.isdir(main_code_dir):
        run.log_code(root=main_code_dir)

    # Add main.py separately if the full execution directory is too large.
    def _only_main(path):
        try:
            return os.path.samefile(path, os.path.join(script_dir, "main.py"))
        except Exception:
            return os.path.abspath(path) == os.path.abspath(os.path.join(script_dir, "main.py"))

    if run is not None:
        run.log_code(root=script_dir, include_fn=_only_main)

    if args.use_wandb and args.train:
        wandb.watch(model)

    # === Outlier-Band Aug init ===
    outlier_aug = None
    if args.outlier_csv:
        outlier_aug = OutlierBandAug(
            outlier_csv=args.outlier_csv,
            prob=args.outlier_prob,
            scale=args.outlier_scale,
            topk=(args.outlier_topk if args.outlier_topk > 0 else None),
            subset_k=args.outlier_subset_k,
            inner_margin=args.inner_margin,
            pre_apodize=args.pre_apodize,
            post_hard_mask=True,  # Keep the region outside the lung at zero.
        )
    # =============================

    # Set the global seed.
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Transforms
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
            # A.Affine(rotate=(-10, 10), scale=(0.8, 1.1),translate_percent=(0.0625, 0.0625),p=0.5),
            # A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.2),
            # A.RandomBrightnessContrast(0.1,0.1,p=0.5),
            # A.ShiftScaleRotate(
            #     shift_limit=0.05,
            #     scale_limit=0.1,
            #     rotate_limit=10,
            #     border_mode=0,
            #     p=0.3
            # ),
            # A.Perspective(scale=(0.02,0.05), p=0.2),
            # A.GaussNoise(
            #     var_limit=(5.0, 20.0),
            #     p=0.3
            # ),        
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # -----------------------------------------
    # Training
    # -----------------------------------------
    if args.train:
        print("=== Training started ===")

        # Dataset
        if args.dataset == "chexpert":
            train_dataset = CheXpertImageDataset(
                split="train", data_path=args.train_data_path, transform=train_transform
            )
            val_dataset = CheXpertImageDataset(
                split="valid", data_path=args.val_data_path, transform=val_transform
            )
        elif args.dataset == "rds":
            train_dataset = RDSImageDataset(
                split="train", data_path=args.train_data_path, transform=train_transform
            )
            val_dataset = RDSImageDataset(
                split="valid", data_path=args.val_data_path, transform=val_transform
            )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init
        )

        mix_criterion = CrossEntropyLoss(size_average=True)
        vanilla_criterion = nn.CrossEntropyLoss()
        val_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_val_acc = 0.0

        for epoch in range(args.num_epochs):
            # ---------------------
            # Train loop
            # ---------------------
            model.train()
            total_loss = 0.0
            train_labels, train_preds, train_probs = [], [], []

            for images, labels, mask in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{args.num_epochs}"):
                images = images.to(device)
                labels_idx = labels.argmax(dim=1).to(device)
                masks = mask.to(device)

                # Outlier-band augmentation
                if outlier_aug is not None:
                    images = outlier_aug(images)

                # Inject the external lung mask into SGN modules.
                lung_mask = masks.to(device)
                if lung_mask.dim() == 3:
                    lung_mask = lung_mask.unsqueeze(1)
                lung_mask = (lung_mask > 0).float()

                target_model = model.module if isinstance(model, nn.DataParallel) else model
                for m in target_model.modules():
                    if isinstance(m, SpectralGatingNetwork):
                        m.set_external_mask(lung_mask)
                # ==========================================

                optimizer.zero_grad()

                do_mix = (
                    (args.mixup or args.stylemix or args.guidedmix or args.lungmix)
                    and random.random() < args.mix_prob
                )

                if do_mix:
                    if args.mixup:
                        images, targets = mixup(images, labels_idx, args.alpha, n_classes=2)
                    elif args.stylemix:
                        images, targets = stylemix(images, labels_idx, args.alpha, n_classes=2, r=args.r)
                    elif args.guidedmix:
                        images, targets = guidedmix(
                            images,
                            labels_idx,
                            n_classes=2,
                            condition=args.condition,
                            saliency_mode=args.saliency_mode,
                            model=model,
                        )
                    else:
                        images, targets = lung_mixup(images, labels_idx, masks, n_classes=2)

                    images, targets = images.to(device), targets.to(device)
                    logits = model(images)
                    loss = mix_criterion(logits, targets)

                    preds = logits.argmax(dim=1)
                    labels_for_metric = targets.argmax(dim=1)
                else:
                    logits = model(images)
                    loss = vanilla_criterion(logits, labels_idx)

                    preds = logits.argmax(dim=1)
                    labels_for_metric = labels_idx

                probs = torch.softmax(logits, dim=1)[:, 1]

                train_labels.extend(labels_for_metric.cpu().tolist())
                train_preds.extend(preds.cpu().tolist())
                train_probs.extend(probs.cpu().tolist())

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_loader)
            train_acc = compute_accuracy(train_labels, train_preds)
            train_f1 = compute_f1(train_labels, train_preds)
            train_auroc = compute_auroc(train_labels, train_probs)
            train_auprc = compute_auprc(train_labels, train_probs)

            print(
                f"Epoch {epoch+1}/{args.num_epochs} - "
                f"Train Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, "
                f"F1: {train_f1:.4f}, AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f}"
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "train_acc": train_acc,
                        "train_f1": train_f1,
                        "train_auroc": train_auroc,
                        "train_auprc": train_auprc,
                        "epoch": epoch + 1,
                    }
                )

            # ---------------------
            # Val loop
            # ---------------------
            model.eval()
            val_loss = 0.0
            val_labels, val_preds, val_probs = [], [], []

            with torch.no_grad():
                for images, labels, mask in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{args.num_epochs}"):
                    images = images.to(device)
                    labels_idx = labels.argmax(dim=1).to(device)
                    masks = mask.to(device)

                    # Inject the external lung mask into SGN modules.
                    lung_mask = masks.to(device)
                    if lung_mask.dim() == 3:
                        lung_mask = lung_mask.unsqueeze(1)
                    lung_mask = (lung_mask > 0).float()

                    target_model = model.module if isinstance(model, nn.DataParallel) else model
                    for m in target_model.modules():
                        if isinstance(m, SpectralGatingNetwork):
                            m.set_external_mask(lung_mask)
                    # =================================

                    logits = model(images)
                    loss = val_criterion(logits, labels_idx)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = torch.argmax(logits, dim=1)

                    val_labels.extend(labels_idx.cpu().tolist())
                    val_preds.extend(preds.cpu().tolist())
                    val_probs.extend(probs.cpu().tolist())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = compute_accuracy(val_labels, val_preds)
            val_f1 = compute_f1(val_labels, val_preds)
            val_auroc = compute_auroc(val_labels, val_probs)
            val_auprc = compute_auprc(val_labels, val_probs)

            print(
                f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, "
                f"Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
                f"AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}"
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "val_loss": avg_val_loss,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "val_auroc": val_auroc,
                        "val_auprc": val_auprc,
                        "epoch": epoch + 1,
                    }
                )

            # ---------------------
            # Save best
            # ---------------------
            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = os.path.join(ckpt_dir, f"best_val_acc_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"New best_val_acc: {best_val_acc:.4f}, saved to {ckpt_path}")

                if args.use_wandb:
                    # Resolve paths safely.
                    try:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                    except NameError:
                        script_dir = os.getcwd()
                    main_py = os.path.join(script_dir, "main.py")
                    main_code_dir = os.path.join(script_dir, "main_code")

                    # Create an artifact, add files/folders, and log it once.
                    model_art = wandb.Artifact(
                        name=f"{args.model_name}_{args.save_model}",
                        type="model",
                        description="Best checkpoint, main.py, and the full ./main_code directory",
                        metadata={"epoch": int(epoch + 1), "val_acc": float(best_val_acc)},
                    )

                    # Checkpoint file
                    model_art.add_file(ckpt_path)

                    # main.py file
                    if os.path.isfile(main_py):
                        model_art.add_file(main_py)
                    else:
                        print(f"[W&B] main.py not found: {main_py}")

                    # Full ./main_code directory
                    if os.path.isdir(main_code_dir):
                        model_art.add_dir(main_code_dir)
                    else:
                        print(f"[W&B] main_code directory not found: {main_code_dir}")

                    wandb.log_artifact(model_art, aliases=["best", f"epoch-{epoch+1}"])

                    print("=== Training finished ===")

    # -----------------------------------------
    # Testing
    # -----------------------------------------
    if args.test:
        print("=== Testing started ===")
        test_only(
            args=args,
            model=model,
            device=device,
            ckpt_dir=ckpt_dir,
            dataset=args.dataset,
            val_transform=val_transform,
            batch_size=args.batch_size,
            use_wandb=args.use_wandb,
        )
        print("=== Testing finished ===")


if __name__ == "__main__":
    main()
