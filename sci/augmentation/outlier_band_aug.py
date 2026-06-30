# sci/augmentation/outlier_band_aug.py
# -*- coding: utf-8 -*-
"""
OutlierBandAug (simple mask with full/block FFT support)

Requirements:
1) Mask the lung region as 1 and the background as 0.
2) Multiply the mask before FFT and again after IFFT to suppress the background.
   This path does not use the older blur/morphology preprocessing.

Notes:
- x is assumed to be float in [0,1]; normalize or clamp in the caller if needed.
  If Normalize(mean,std) has already been applied, adjust nz_threshold accordingly.
- The mask is derived from non-zero pixels: g > nz_threshold -> 1, else 0.
- If block is provided, use block_fft_ifft; otherwise use full_fft_ifft.
"""

from __future__ import annotations
import csv
import os
import random
from typing import List, Optional

import torch

# Project FFT functions.
from sci.augmentation.fft_transform import full_fft_ifft as _full_fft_ifft
from sci.augmentation.fft_transform import block_fft_ifft as _block_fft_ifft


# ----------------------------- CSV loader -----------------------------
def _load_band_ids(csv_path: str) -> List[int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"outlier_csv not found: {csv_path}")
    out: List[int] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                out.append(int(r["band_id"]))
            except Exception:
                continue
    if not out:
        raise ValueError(f"'{csv_path}'  did not contain any readable band_id values.")
    return out


# ----------------------------- full_fft_ifft safe wrapper -----------------------------
def _safe_full_fft_ifft(
    x: torch.Tensor,
    band_ids: List[int],
    scale: float,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """
    Call full_fft_ifft while tolerating minor signature differences.
    x: accepts (C,H,W) or (1,C,H,W).
    """
    # 1) bands_to_scale keyword.
    try:
        return _full_fft_ifft(
            x,
            bands_to_scale=band_ids,
            scale_factor=scale,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
    except TypeError:
        pass
    # 2) positional arguments.
    try:
        return _full_fft_ifft(x, band_ids, scale, clamp_min, clamp_max)
    except TypeError:
        pass
    # 3) indices_to_scale keyword.
    try:
        return _full_fft_ifft(
            x,
            indices_to_scale=band_ids,
            scale_factor=scale,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
    except TypeError:
        # 4) Handle implementations that require a batch dimension.
        try:
            xb = x.unsqueeze(0)  # (1,C,H,W)
            yb = _full_fft_ifft(
                xb,
                bands_to_scale=band_ids,
                scale_factor=scale,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
            return yb.squeeze(0)
        except Exception as e:
            raise TypeError(f"full_fft_ifft call failed: {e}")


# ----------------------------- block_fft_ifft safe wrapper -----------------------------
def _safe_block_fft_ifft(
    x: torch.Tensor,
    band_ids: List[int],
    scale: float,
    clamp_min: float,
    clamp_max: float,
    block: int,
) -> torch.Tensor:
    """
    Call block_fft_ifft while tolerating minor signature differences.
    x: accepts (C,H,W) or (1,C,H,W).
    """
    # 1) keyword arguments.
    try:
        return _block_fft_ifft(
            x,
            block=block,
            bands_to_scale=band_ids,
            scale_factor=scale,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
    except TypeError:
        pass
    # 2) Positional arguments for compatibility with older versions.
    try:
        return _block_fft_ifft(x, block, band_ids, scale, clamp_min, clamp_max)
    except TypeError as e:
        # 3) Handle implementations that require a batch dimension.
        try:
            xb = x.unsqueeze(0)  # (1,C,H,W)
            yb = _block_fft_ifft(
                xb,
                block=block,
                bands_to_scale=band_ids,
                scale_factor=scale,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
            return yb.squeeze(0)
        except Exception as e2:
            raise TypeError(f"block_fft_ifft call failed: {e} / {e2}")


# ----------------------------- Augmenter -----------------------------
class OutlierBandAug:
    """
    Outlier-band scaling augmenter using a simple binary mask.

    Args:
        outlier_csv: CSV containing a band_id column
            - full FFT: 0..(H*W-1), assuming unshifted indexing
            - block FFT: 0..(block*block-1), where band_id = fy*block + fx
        prob: application probability
        scale: scale_factor
        clamp_min/max: output clamp range
        topk: use only the first k CSV rows; 0 or None uses all rows
        subset_k: randomly select subset_k band IDs on each call; 0 uses all IDs
        nz_threshold: mask threshold, default 1/255; g > threshold -> 1
        block: enables block FFT mode when set, for example 8; None uses full FFT.
        inner_margin/pre_apodize/post_hard_mask/mask_mode: kept for compatibility and unused here
    """
    def __init__(
        self,
        outlier_csv: str,
        prob: float = 0.5,
        scale: float = 2.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        topk: Optional[int] = None,
        subset_k: int = 0,
        nz_threshold: float = 1.0 / 255.0,
        block: Optional[int] = None,
        # Compatibility arguments; ignored here.
        inner_margin: int = 0,
        pre_apodize: float = 0.0,
        post_hard_mask: bool = True,
        mask_mode: str = "binary_lung",
    ):
        band_ids = _load_band_ids(outlier_csv)
        if topk is not None and topk > 0:
            band_ids = band_ids[:topk]
        self.band_ids: List[int] = band_ids

        self.prob = float(prob)
        self.scale = float(scale)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.subset_k = int(subset_k)
        self.nz_threshold = float(nz_threshold)
        self.block = int(block) if block else None  # None -> full FFT mode.

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) or (C,H,W) float
        return: same shape
        """
        if self.prob < 1.0 and torch.rand(1).item() >= self.prob:
            return x

        is_batched = (x.dim() == 4)
        if not is_batched:
            x = x.unsqueeze(0)  # -> (1,C,H,W)

        B, C, H, W = x.shape
        out = torch.empty_like(x)

        for i in range(B):
            xi = x[i]  # (C,H,W)

            # 1) Build a binary lung-like mask from the input using the channel mean.
            if C > 1:
                g = xi.mean(dim=0, keepdim=True)[0]  # [H,W]
            else:
                g = xi[0]  # [H,W]
            hard = (g > self.nz_threshold).to(dtype=xi.dtype, device=xi.device).unsqueeze(0)  # [1,H,W]

            # Avoid the edge case in which the mask is entirely zero.
            if torch.count_nonzero(hard) == 0:
                out[i] = xi
                continue

            # 2) Multiply the mask before FFT; soft and hard masks are identical here.
            xi_in = (xi * hard).clamp(self.clamp_min, self.clamp_max)

            # Select bands; use a random subset when subset_k > 0.
            if self.subset_k and self.subset_k < len(self.band_ids):
                band_ids = random.sample(self.band_ids, self.subset_k)
            else:
                band_ids = self.band_ids

            # 3) Scale bands using block FFT or full FFT.
            if self.block:
                yi = _safe_block_fft_ifft(
                    xi_in, band_ids, self.scale, self.clamp_min, self.clamp_max, self.block
                )
            else:
                yi = _safe_full_fft_ifft(
                    xi_in, band_ids, self.scale, self.clamp_min, self.clamp_max
                )

            # 4) Multiply the same mask after IFFT to suppress outside-mask regions.
            yi = (yi * hard).clamp(self.clamp_min, self.clamp_max)

            out[i] = yi

        return out if is_batched else out.squeeze(0)


def build_lung_mask(x: torch.Tensor, nz_threshold: float = 1.0/255.0):
    """
    Input x: (1,H,W) or (C,H,W) float
    Return hard and soft masks, both [1,H,W] float32.
    - hard: 1 where input exceeds nz_threshold, otherwise 0
    - soft: identical to hard, without blur
    """
    if x.dim() == 2:
        g = x
    elif x.dim() == 3:
        C, H, W = x.shape
        g = x.mean(dim=0) if C > 1 else x[0]
    else:
        raise ValueError("x must be (C,H,W) or (H,W)")
    hard = (g > nz_threshold).to(dtype=torch.float32, device=x.device).unsqueeze(0)
    soft = hard.clone()
    return hard, soft
