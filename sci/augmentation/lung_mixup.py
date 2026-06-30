import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.ndimage import label
from typing import Tuple

def onehot(label: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.zeros(label.size(0), n_classes, device=label.device).scatter_(
        1, label.view(-1, 1), 1)

def split_left_right(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    mask: (H, W) 이진 numpy array
    returns: (left_mask, right_mask) 둘 다 bool array
    """
    lab, n = label(mask)
    if n < 2:
        raise RuntimeError("폐가 두 덩어리로 분리되지 않습니다.")
    # 가장 큰 두 컴포넌트만 골라서
    areas = [(lab==i).sum() for i in range(1, n+1)]
    top2 = np.argsort(areas)[-2:] + 1
    # 컴포넌트별 중심 x 좌표
    centers = [(np.nonzero(lab==c)[1].mean()) for c in top2]
    # 작은 x가 왼쪽, 큰 x가 오른쪽
    left_i, right_i = (0,1) if centers[0] < centers[1] else (1,0)
    return (lab==top2[left_i]), (lab==top2[right_i])

def lung_mixup(
    imgs: torch.Tensor,     # (B, C, H, W), float [0..1]
    labels: torch.Tensor,   # (B,) 정수 labels
    masks: torch.Tensor,    # (B, H, W), {0,1} 이진 마스크
    n_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    좌/우 폐 마스크로만 mixup 하는 함수.
    returns:
      mixed_imgs:    (B, C, H, W)
      mixed_targets: (B, n_classes) soft labels
    """
    B, C, H, W = imgs.shape
    device = imgs.device

    # 1) 랜덤 페어링
    idx = torch.randperm(B, device=device)
    imgs2   = imgs[idx]
    labels2 = labels[idx]
    masks2  = masks[idx]

    # 2) one-hot
    y1 = onehot(labels, n_classes)
    y2 = onehot(labels2, n_classes)

    mixed = torch.empty_like(imgs)
    lam   = torch.empty(B, 1, device=device)

    for i in range(B):
        # numpy mask로 분리
        
        m1 = masks[i].cpu().numpy().astype(bool)
        m2 = masks2[i].cpu().numpy().astype(bool)

        l1, r1 = split_left_right(m1)  # (H, W)
        l2, r2 = split_left_right(m2)

        # tensor로 변환
        l1_t = torch.from_numpy(l1.astype(np.float32)).to(device)  # (H, W)
        r2_t = torch.from_numpy(r2.astype(np.float32)).to(device)

        # pixel-wise mix
        mask = l1_t.unsqueeze(0).expand(C, -1, -1)   # (C, H, W)
        invm = r2_t.unsqueeze(0).expand(C, -1, -1)   # (C, H, W)
        mixed[i] = imgs[i] * mask + imgs2[i] * invm

        # lam = mask 비율
        lam_i = mask.mean()                         # float
        lam[i] = lam_i

    # 3) soft label: lam·y1 + (1-lam)·y2
    mixed_targets = lam * y1 + (1 - lam) * y2

    return mixed, mixed_targets
