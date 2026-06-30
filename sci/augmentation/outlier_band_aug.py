# sci/augmentation/outlier_band_aug.py
# -*- coding: utf-8 -*-
"""
OutlierBandAug (단순 마스크 + full/block FFT 지원)

요구사항:
1) 입력 이미지의 폐 영역만 1(배경 0)로 마스크
2) FFT 전 입력에 마스크 곱 → IFFT 후에도 동일 마스크 곱으로 배경 제거
   (블러/모폴로지 등 기존 전처리 폐기)

주의:
- 입력 x는 float [0,1] 범위를 가정 (필요 시 caller에서 normalize/clamp).
  (만약 Normalize(mean,std)를 적용한 텐서라면 nz_threshold를 상황에 맞게 조정하세요.)
- mask은 x의 (non-zero) 픽셀을 기준으로 생성: g > nz_threshold → 1, else 0
- block 매개변수가 주어지면 block_fft_ifft 경로, 아니면 full_fft_ifft 경로로 동작
"""

from __future__ import annotations
import csv
import os
import random
from typing import List, Optional

import torch

# 프로젝트 FFT 함수
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
        raise ValueError(f"'{csv_path}' 에서 band_id를 한 개도 읽지 못했습니다.")
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
    full_fft_ifft의 시그니처 변화에 안전하게 호출.
    x: (C,H,W) 또는 (1,C,H,W)도 허용
    """
    # 1) bands_to_scale 키워드
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
    # 2) 위치 인자
    try:
        return _full_fft_ifft(x, band_ids, scale, clamp_min, clamp_max)
    except TypeError:
        pass
    # 3) indices_to_scale 키워드
    try:
        return _full_fft_ifft(
            x,
            indices_to_scale=band_ids,
            scale_factor=scale,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
    except TypeError:
        # 4) 배치 차원만 받는 구현 대응
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
            raise TypeError(f"full_fft_ifft 호출 실패: {e}")


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
    block_fft_ifft의 시그니처 변화에 안전하게 호출.
    x: (C,H,W) 또는 (1,C,H,W)도 허용
    """
    # 1) keyword 인자
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
    # 2) 위치 인자 (구버전 호환)
    try:
        return _block_fft_ifft(x, block, band_ids, scale, clamp_min, clamp_max)
    except TypeError as e:
        # 3) 배치 차원만 받는 구현 대응
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
            raise TypeError(f"block_fft_ifft 호출 실패: {e} / {e2}")


# ----------------------------- Augmenter -----------------------------
class OutlierBandAug:
    """
    이상치 밴드 강조 Augmenter (단순 이진 마스크 방식).

    Args:
        outlier_csv: band_id 열이 있는 CSV
            - full FFT일 때: 0..(H*W-1) (unshifted 가정)
            - block FFT일 때: 0..(block*block-1) (band_id = fy*block + fx)
        prob: 적용 확률
        scale: scale_factor
        clamp_min/max: 결과 clamp 범위
        topk: CSV 상위 k개만 사용 (0 또는 None 이면 전체)
        subset_k: 호출마다 band_ids 중 무작위 subset_k개만 골라 적용(0이면 전체)
        nz_threshold: 마스크 임계값 (기본 1/255). g > threshold → 1
        block: 블록 크기 지정 시 block FFT 모드로 동작 (예: 8). None이면 full FFT.
        inner_margin/pre_apodize/post_hard_mask/mask_mode: 호환성 유지용(미사용)
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
        # 아래는 호환용(무시)
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
        self.block = int(block) if block else None  # None → full FFT 모드

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) 또는 (C,H,W) float
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

            # 1) 입력으로부터 폐 이진 마스크 만들기 (채널 평균 기준)
            if C > 1:
                g = xi.mean(dim=0, keepdim=True)[0]  # [H,W]
            else:
                g = xi[0]  # [H,W]
            hard = (g > self.nz_threshold).to(dtype=xi.dtype, device=xi.device).unsqueeze(0)  # [1,H,W]

            # 마스크가 전부 0인 엣지 케이스 방지
            if torch.count_nonzero(hard) == 0:
                out[i] = xi
                continue

            # 2) FFT 전에 마스크 곱 (soft=hard 동일)
            xi_in = (xi * hard).clamp(self.clamp_min, self.clamp_max)

            # 밴드 선택 (subset_k>0 이면 무작위 부분집합)
            if self.subset_k and self.subset_k < len(self.band_ids):
                band_ids = random.sample(self.band_ids, self.subset_k)
            else:
                band_ids = self.band_ids

            # 3) 강조: block 모드면 block_fft_ifft, 아니면 full_fft_ifft
            if self.block:
                yi = _safe_block_fft_ifft(
                    xi_in, band_ids, self.scale, self.clamp_min, self.clamp_max, self.block
                )
            else:
                yi = _safe_full_fft_ifft(
                    xi_in, band_ids, self.scale, self.clamp_min, self.clamp_max
                )

            # 4) IFFT 후에도 동일 마스크 곱으로 폐 외 영역 완전 제거
            yi = (yi * hard).clamp(self.clamp_min, self.clamp_max)

            out[i] = yi

        return out if is_batched else out.squeeze(0)


def build_lung_mask(x: torch.Tensor, nz_threshold: float = 1.0/255.0):
    """
    입력 x: (1,H,W) 또는 (C,H,W) float
    반환: hard, soft (둘 다 [1,H,W], float32)
    - hard: 입력에서 nz_threshold보다 큰 부분을 1, 나머지는 0
    - soft: hard와 동일 (블러 없이 그대로 사용)
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
