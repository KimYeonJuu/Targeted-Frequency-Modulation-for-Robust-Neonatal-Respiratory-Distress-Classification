import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import net_mixup
import torchvision.transforms.functional as TF

from .pairing import onecycle_cover as pairing
from .spectral_residual import SpectralResidual
sr = SpectralResidual()

def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes, device=label.device).scatter_(
        1, label.view(-1, 1), 1)

##################################################
# MixUp
##################################################

def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0), device=data.device)
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(data.device)
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets

##################################################
# StyleMix
##################################################

def stylemix(data, targets, alpha, n_classes, r):
    """
    StyleMixup implementation aligned with the official StyleMix train.py.
    Args:
        data (Tensor): (B, C, H, W)
        targets (LongTensor): (B,)
        alpha (float): beta-distribution parameter
        n_classes (int): number of classes
        r (float): content/style loss weight (args.r)
    Returns:
        mixed (Tensor): (B, C, H, W) augmented images
        mixed_targets (Tensor): (B, n_classes) mixed soft labels
    """
    torch.cuda.empty_cache()
    
    decoder   = net_mixup.decoder
    vgg       = net_mixup.vgg
    network_E = net_mixup.Net_E(vgg)
    network_D = net_mixup.Net_D(vgg, decoder)
            
    stylemix_dir = os.environ.get("STYLEMIX_MODEL_DIR", "stylemix_model")
    vgg.load_state_dict(torch.load(os.path.join(stylemix_dir, "vgg_normalised.pth")))
    decoder.load_state_dict(torch.load(os.path.join(stylemix_dir, "decoder.pth.tar")))
    
    vgg = nn.Sequential(*list(vgg.children())[:31]).cuda().eval()
    decoder = decoder.cuda().eval()
    network_E = torch.nn.DataParallel(network_E).cuda().eval()
    network_D = torch.nn.DataParallel(network_D).cuda().eval()

    B, C, H, W = data.size()
    # 1) shuffle
    idx = torch.randperm(B, device=data.device)
    target1 = targets
    target2 = targets[idx]
    # one-hot
    t1 = onehot(target1, n_classes)
    t2 = onehot(target2, n_classes)

    # 2) upsample to 224x224
    up224 = nn.Upsample(size=(224,224), mode='bilinear', align_corners=False)
    x1 = up224(data)
    x2 = x1[idx]

    # 3) sample mixing ratios
    rc = np.random.beta(alpha, alpha)
    rs = np.random.beta(alpha, alpha)

    # 4) encode + decode (no gradient)
    with torch.no_grad():
        f1 = network_E(x1)
        mixed224 = network_D(f1, f1[idx], rc, rs)

    # 5) downsample back to original size
    down = nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)
    mixed = down(mixed224)

    # 6) Compute soft labels.
    # Weighted combination of content-loss and style-loss distributions.
    content_dist = rc * t1 + (1 - rc) * t2
    style_dist   = rs * t1 + (1 - rs) * t2
    mixed_targets = r * content_dist + (1 - r) * style_dist

    return mixed, mixed_targets

##################################################
# GuidedMix
##################################################

def pairing_wrapper(sc, condition='random', distance_metric='l2'):
    """
    sc: (B,H,W) real-valued saliency-map tensor
    condition: 'random' or 'greedy'
    returns: index array of length B
    """
    B, H, W = sc.shape

    if condition == 'greedy':
        # 1) Compute the distance matrix (B x B).
        #    distance_function flattens maps and computes L2/cosine-style distances.
        X = distance_function(sc, sc, distance_metric).cpu().numpy()
        # 2) Pass the matrix to onecycle_cover to compute pairing.
        sorted_indices = onecycle_cover(X)
    else:
        # Random shuffle.
        sorted_indices = np.random.permutation(B)

    return sorted_indices

def compute_grad_saliency(images: torch.Tensor, model: nn.Module, targets: torch.Tensor) -> torch.Tensor:
    """
    Generate saliency maps from gradients.
    """
    images_var = images.clone().detach().requires_grad_(True).cuda()
    targets_cuda = targets.cuda()
    model.eval()
    outputs = model(images_var)
    loss = F.cross_entropy(outputs, targets_cuda)
    loss.backward()
    sal = images_var.grad.abs().max(dim=1)[0]  # (B, H, W)
    return sal

def distance_function(a: torch.Tensor, b: torch.Tensor=None, distance_metric: str='l2') -> torch.Tensor:
    """
    Return the pairwise distance matrix between flattened saliency maps a and b.
    Only 'l2' is currently supported.
    Args:
      a: Tensor (B, H, W)
      b: Tensor (B, H, W) or None (same as a)
      distance_metric: 'l2'
    Returns:
      dist: Tensor (B, B)
    """
    if b is None:
        b = a
    B, H, W = a.shape
    a_flat = a.reshape(B, -1)
    b_flat = b.reshape(B, -1)
    if distance_metric == 'l2':
        # Use torch.cdist to compute pairwise L2 distances.
        return torch.cdist(a_flat, b_flat, p=2)
    else:
        raise NotImplementedError(f"Distance metric '{distance_metric}' not implemented")

def guidedmix(data: torch.Tensor,
              targets: torch.Tensor,
              n_classes: int,
              condition: str = 'random',
              saliency_mode: str = 'spectral',
              model: nn.Module = None,
              grad: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    """
    Unified GuidedMixup function. Depending on saliency_mode:
    - 'spectral': SpectralResidual-based saliency
    - 'grad': gradient-based saliency; requires model
    Args:
      data         (Tensor): (B, C, H, W) input batch
      targets      (Tensor): (B,) integer labels
      n_classes    (int):    number of classes
      condition    (str):    'random' or 'greedy'
      saliency_mode(str):    'spectral' or 'grad'
      model        (nn.Module): required for gradient mode
      grad         (Tensor): (B, H, W) externally computed saliency
    Returns:
      mixed        (Tensor): (B, C, H, W) mixed images
      mixed_tgt    (Tensor): (B, n_classes) soft-label
    """
    B, C, H, W = data.shape

    # 1) Generate or assign the saliency map.
    if saliency_mode == 'grad':
        # Gradient-based saliency.
        assert model is not None, "model must be provided to compute gradient saliency"
        if grad is None:
            data_var = data.clone().detach().requires_grad_(True)
            outputs = model(data_var)
            loss = F.cross_entropy(outputs, targets.cuda())
            loss.backward()
            sc = data_var.grad.abs().max(dim=1)[0]
        else:
            sc = grad
    else:
        # Spectral residual-based saliency.
        sc = sr.transform_spectral_residual(data)

    # Post-process with blur and normalization.
    sc = TF.gaussian_blur(sc.unsqueeze(1), kernel_size=(7,7), sigma=(3,3)).squeeze(1)
    sc = sc / sc.sum(dim=[-1,-2], keepdim=True)

    # 2) Pairing.
    if condition == 'greedy':
        # Distance matrix plus onecycle_cover.
        X = distance_function(sc, sc, 'l2').cpu().numpy()
        idx = pairing(X)   # onecycle_cover alias
    else:
        idx = np.random.permutation(B)

    # 3) Fetch paired data.
    data_b = data[idx]
    sc_b   = sc[idx]

    # 4) Pixel-wise mixing mask.
    norm_sc = sc / (sc + sc_b).detach()  # (B, H, W)
    mask    = norm_sc.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, H, W)

    # 5) Mix images.
    mixed = mask * data + (1 - mask) * data_b

    # 6) Mix soft labels.
    t1  = onehot(targets, n_classes)
    t2  = onehot(targets[idx], n_classes)
    lam = norm_sc.mean(dim=[-1,-2]).unsqueeze(-1)
    mixed_tgt = lam * t1 + (1 - lam) * t2

    return mixed, mixed_tgt

##################################################
# HalfLungMixup
##################################################

def lung_half_mixup(
    data: torch.Tensor,      # (B, C, H, W)
    targets: torch.Tensor,   # (B,) integer labels
    n_classes: int
) -> (torch.Tensor, torch.Tensor):
    """
    Randomly pair samples within the batch, then create:
      mix1 = [A_left | B_right]
      mix2 = [B_left | A_right]
    two new images.
    Soft labels are mixed evenly with weights (0.5, 0.5).

    Returns:
      mixed    : (2*B, C, H, W)
      mix_tgts : (2*B, n_classes)
    """
    B, C, H, W = data.shape
    device = data.device

    # 1) Random pairing.
    idx = torch.randperm(B, device=device)
    data_b = data[idx]
    tgt_b  = targets[idx]

    # 2) Prepare soft labels (0.5*A + 0.5*B).
    y1 = onehot(targets, n_classes)
    y2 = onehot(tgt_b,    n_classes)
    mix_tgt = 0.5 * y1 + 0.5 * y2   # (B, n_classes)

    # 3) Split left and right halves.
    mid = W // 2
    A_left  = data[:, :, :, :mid]    # (B,C,H,mid)
    A_right = data[:, :, :, mid:]    # (B,C,H,W-mid)
    B_left  = data_b[:, :, :, :mid]
    B_right = data_b[:, :, :, mid:]

    # 4) Create two mixed batches.
    mix1 = torch.cat([A_left,  B_right], dim=3)  # (B,C,H,W)
    mix2 = torch.cat([B_left,  A_right], dim=3)  # (B,C,H,W)

    # 5) Expand the batch.
    mixed    = torch.cat([mix1,    mix2],    dim=0)    # (2B, C, H, W)
    targets = torch.cat([mix_tgt, mix_tgt], dim=0)    # (2B, n_classes)

    return mixed, targets

def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)
    
    
    
    
    
    
    
