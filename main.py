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


# 환경 변수
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
    # 최신 체크포인트 로드
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"체크포인트({ckpt_dir}/*.pth)를 찾을 수 없습니다.")
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    print(f"[Test] Loaded checkpoint: {latest_ckpt}")

    # 테스트 데이터셋
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

    # 평가
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_labels, test_preds, test_probs = [], [], []

    with torch.no_grad():
        for images, labels, mask in tqdm(test_loader, desc="[Test]"):
            images = images.to(device)
            labels_idx = labels.argmax(dim=1).to(device)
            masks = mask.to(device)

            # === SGN 외부 마스크 주입 ===
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

    # 경로
    parser.add_argument("--train_data_path", type=str, required=True, help="학습 데이터 경로")
    parser.add_argument("--val_data_path", type=str, required=True, help="검증 데이터 경로")
    parser.add_argument("--test_data_path", type=str, required=True, help="테스트 데이터 경로")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="체크포인트 저장 디렉토리")
    parser.add_argument("--save_model", type=str, default="model", help="모델 저장 이름")

    # 실행/환경
    parser.add_argument("--project_name", type=str, default="SCI-RDS", help="Wandb 프로젝트 이름")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_parallel", action="store_true", help="DataParallel 사용")
    parser.add_argument("--gpu", type=str, default="0", help='사용할 GPU 번호 (예: "0,1")')

    # 모델/데이터
    parser.add_argument("--model_name", type=str, required=True, help="timm 모델 이름 (e.g., tf_efficientnet_b0)")
    parser.add_argument("--dataset", type=str, required=True, help="데이터셋 종류")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("--num_epochs", type=int, default=30, help="학습 에폭 수")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--freeze_backbone",
        type=str,
        default="OFF",
        choices=["ON", "OFF"],
        help='"ON" → backbone freeze, "OFF" → full fine-tuning',
    )

    # SGN 블록/삽입 스펙
    parser.add_argument(
        "--block-selection",
        "--sb",
        dest="sb_spec",
        type=str,
        default="",
        help="Spectral Block 교체 지정(1-기반). 예) '1:1,1:2;2:1' or '1:*;2:1-2'",
    )

    # === SGN 하이퍼파라미터 ===
    parser.add_argument("--sgn_mode", choices=["grid", "block"], default="grid", help="SGN 주파수 처리 모드")
    parser.add_argument("--sgn_grid_size", type=str, default="16x16", help="grid 모드 고정 격자 크기(HxW)")
    parser.add_argument("--sgn_tau", type=float, default=0.25, help="소프트 게이팅 온도")
    parser.add_argument("--sgn_amp", type=float, default=1.0, help="강조 강도 (scale = 1 + amp * gate)")
    parser.add_argument("--sgn_block", type=str, default="16x16", help="block 모드 블록 크기(HxW)")
    parser.add_argument("--sgn_stride", type=str, default="same", help='block 모드 stride(HxW), "same"이면 block과 동일')

    # === SGN outlier 선택 하이퍼파라미터 ===
    parser.add_argument("--sgn_ratio_thresh", type=float, default=0.02, help="에너지 비율 임계값")
    parser.add_argument("--sgn_outlier_topk", type=int, default=0, help="상위 K 주파수만 선택(0 비활성)")
    parser.add_argument("--sgn_min_radius", type=int, default=1, help="저주파 제외 반경")
    parser.add_argument("--sgn_max_radius", type=int, default=-1, help="최대 반경 제한(-1 비활성)")
    parser.add_argument("--sgn_mask_off", action="store_true", help="outlier 마스킹 비활성")

    # 부가 모듈/증강
    parser.add_argument("--use_cbam", action="store_true", help="CBAM 사용")
    parser.add_argument("--use_gtfe", action="store_true", help="GTFE 사용")

    parser.add_argument("--grid_shuffle_2_0_5", action="store_true", help="Grid Shuffle 사용")
    parser.add_argument("--grid_shuffle_3_0_5", action="store_true", help="Grid Shuffle 사용")
    parser.add_argument("--grid_shuffle_2_1_0", action="store_true", help="Grid Shuffle 사용")
    parser.add_argument("--grid_shuffle_3_1_0", action="store_true", help="Grid Shuffle 사용")

    parser.add_argument("--mixup", action="store_true", help="Mixup 사용")
    parser.add_argument("--stylemix", action="store_true", help="StyleMix 사용")
    parser.add_argument("--guidedmix", action="store_true", help="GuidedMix 사용")
    parser.add_argument("--lungmix", action="store_true", help="LungMix 사용")
    parser.add_argument("--mix_prob", type=float, default=0.5, help="Mix 계열 적용 확률")
    parser.add_argument("--alpha", type=float, default=1.0, help="Mixup alpha")
    parser.add_argument("--r", type=float, default=0.5, help="StyleMix r")
    parser.add_argument("--condition", type=str, default="greedy", help="GuidedMix condition")
    parser.add_argument("--saliency_mode", type=str, default="grad", help="GuidedMix saliency mode")

    parser.add_argument("--fft_aug", action="store_true", help="8×8 block FFT augment 적용 여부")
    parser.add_argument("--fft_aug_prob", type=float, default=0.5, help="FFT augment 적용 확률 (0~1)")
    parser.add_argument("--stride_1", action="store_true", help="Stride 1 사용")

    # Outlier-Band Aug
    parser.add_argument("--outlier_csv", type=str, default="", help="이상치 밴드 CSV 경로")
    parser.add_argument("--outlier_prob", type=float, default=0.5)
    parser.add_argument("--outlier_scale", type=float, default=2.0)
    parser.add_argument("--outlier_topk", type=int, default=0)
    parser.add_argument("--outlier_subset_k", type=int, default=0)
    parser.add_argument("--inner_margin", type=int, default=1)
    parser.add_argument("--pre_apodize", type=float, default=1.0)
    parser.add_argument("--post_hard_mask", action="store_true", help="IFFT 후 배경 0 유지")

    args = parser.parse_args()

    # === SGN 기본값 설정 ===
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

    # GPU 선택
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")

    # 기본 실행: train
    if not args.train and not args.test:
        args.train = True

    set_seed(args.seed)

    # 체크포인트 경로
    ckpt_root = args.checkpoint_dir
    ckpt_dir = os.path.join(ckpt_root, args.model_name, args.save_model)
    print(f"ckpt_dir : {ckpt_dir}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 디바이스/프리즈 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_flag = args.freeze_backbone.upper() == "ON"

    # 모델 생성
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
        print(f"[Info] DataParallel 사용, GPU: {args.gpu}")
        model = nn.DataParallel(model)

    model.to(device)

    # wandb 초기화
    run = None
    if args.use_wandb and (args.train or args.test):
        run = wandb.init(
        project=args.project_name,
        name=args.save_model,
        config=vars(args),
        reinit=True,
        save_code=False,   # 자동 스냅샷 대신 아래에서 명시적으로 올림
    )
        
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # main.py가 있는 폴더
    except NameError:
        script_dir = os.getcwd()

    main_code_dir = os.path.join(script_dir, "main_code")

    # 폴더 통째로
    if os.path.isdir(main_code_dir):
        run.log_code(root=main_code_dir)

    # main.py만 따로 (실행 디렉토리 전체가 너무 크면 include_fn으로 main.py만 올림)
    def _only_main(path):
        try:
            return os.path.samefile(path, os.path.join(script_dir, "main.py"))
        except Exception:
            return os.path.abspath(path) == os.path.abspath(os.path.join(script_dir, "main.py"))

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
            post_hard_mask=True,  # 폐 외부 0 유지
        )
    # =============================

    # 시드 고정(글로벌) - 원본 유지
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 변환
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
        print("=== Training 시작 ===")

        # 데이터셋
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

                # === SGN 외부(원본기반) 폐 마스크 주입 ===
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

                    # === SGN 외부 마스크 주입 ===
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
                    # 경로 안전하게 확보
                    try:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                    except NameError:
                        script_dir = os.getcwd()
                    main_py = os.path.join(script_dir, "main.py")
                    main_code_dir = os.path.join(script_dir, "main_code")

                    # 아티팩트 생성 → 파일/폴더 추가 → 한 번만 로그
                    model_art = wandb.Artifact(
                        name=f"{args.model_name}_{args.save_model}",
                        type="model",
                        description="best checkpoint + main.py + ./main_code/ 전체",
                        metadata={"epoch": int(epoch + 1), "val_acc": float(best_val_acc)},
                    )

                    # 체크포인트 파일
                    model_art.add_file(ckpt_path)

                    # main.py 파일
                    if os.path.isfile(main_py):
                        model_art.add_file(main_py)
                    else:
                        print(f"[W&B] main.py를 찾지 못했습니다: {main_py}")

                    # ./main_code 폴더 전체
                    if os.path.isdir(main_code_dir):
                        model_art.add_dir(main_code_dir)
                    else:
                        print(f"[W&B] main_code 폴더를 찾지 못했습니다: {main_code_dir}")

                    wandb.log_artifact(model_art, aliases=["best", f"epoch-{epoch+1}"])

                    print("=== Training 완료 ===")

    # -----------------------------------------
    # Testing
    # -----------------------------------------
    if args.test:
        print("=== Testing 시작 ===")
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
        print("=== Testing 완료 ===")


if __name__ == "__main__":
    main()
