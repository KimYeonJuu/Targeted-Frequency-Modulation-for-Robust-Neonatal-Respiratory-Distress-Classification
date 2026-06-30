# fft_transform.py

import torch
from typing import List, Optional, Union

def block_fft_ifft(
    x: torch.Tensor,
    block: int = 8,
    bands_to_scale: Optional[List[int]] = None,
    scale_factor: float = 2.0,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    입력 텐서 x에 대해 8×8 블록별 FFT → 지정 밴드 강조 → IFFT → 클램프 → 재조합
    - x: [C,H,W] 또는 [B,C,H,W]
    - C: 채널 수 (보통 1 또는 3)
    - B: 배치 크기
    """
    if bands_to_scale is None:
        bands_to_scale = [8,16,24,32,40,48,56]

    # (1) 만약 [C,H,W] 형태라면 [1,C,H,W] 로 바꿔주기
    is_batch = (x.dim() == 4)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() != 4:
        raise ValueError(f"지원하지 않는 입력 차원: {x.shape}")

    B, C, H, W = x.shape
    nh, nw = H // block, W // block

    out = torch.zeros_like(x)
    for b in range(B):
        for c in range(C):
            single = x[b, c:c+1]  # shape [1,H,W]

            # 원본 범위 기억
            orig_min, orig_max = single.min(), single.max()
            # 0~1 사이로 클램핑
            single = single.clamp(clamp_min, clamp_max)

            # 블록 분할
            blocks = (
                single
                .unfold(1, block, block)
                .unfold(2, block, block)
                .permute(1,2,0,3,4)  # [nh, nw, 1, block, block]
            )[..., 0, :, :]         # [nh, nw, block, block]

            # FFT → 지정 밴드 강조
            B_fft = torch.fft.fft2(blocks)  # complex [nh,nw,block,block]
            B_flat = B_fft.reshape(nh, nw, block*block)
            for idx in bands_to_scale:
                B_flat[:,:,idx] *= scale_factor
            B_fft = B_flat.view(nh, nw, block, block)

            # IFFT → 실수부
            B_ifft = torch.fft.ifft2(B_fft).real  # [nh,nw,block,block]

            # 블록 재조합
            recon = torch.zeros_like(single)
            for i in range(nh):
                for j in range(nw):
                    recon[0,
                          i*block:(i+1)*block,
                          j*block:(j+1)*block] = B_ifft[i,j]

            # 원본 범위 벗어난 값만 clamp
            recon = recon.clamp(orig_min.item(), orig_max.item())

            out[b, c] = recon

    # (2) 원래 차원으로 복원
    if not is_batch:
        out = out.squeeze(0)  # [C,H,W]
    return out

def full_fft_ifft(
    x: torch.Tensor,
    bands_to_scale: Optional[List[int]] = None,
    scale_factor: float = 2.0,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    블록 분할 없이 전체 이미지에 대해 FFT → 선택된 주파수 인덱스 강조 → IFFT → 클램프
    - x: [C,H,W] 또는 [B,C,H,W]
    - bands_to_scale: flatten된 FFT (H*W)에서 강조할 빈 인덱스 리스트
    - scale_factor: 해당 밴드에 곱할 계수
    - clamp_min/max: 입력을 먼저 클램핑할 범위 (기본 0~1)
    반환: 원래 입력과 동일한 shape, 실수부만 사용
    """
    if bands_to_scale is None:
        bands_to_scale = []  # 기본 강조 없음

    # 배치 처리 정리
    is_batch = (x.dim() == 4)
    if x.dim() == 3:
        x = x.unsqueeze(0)  # [1,C,H,W]
    elif x.dim() != 4:
        raise ValueError(f"지원하지 않는 입력 차원: {x.shape}")

    B, C, H, W = x.shape
    out = torch.zeros_like(x)

    for b in range(B):
        for c in range(C):
            single = x[b, c:c+1]  # [1,H,W]

            # 원본 범위 저장
            orig_min, orig_max = single.min(), single.max()

            # 입력 클램핑
            single = single.clamp(clamp_min, clamp_max)

            # FFT2 전체 이미지
            # torch.fft.fft2 expects at least 2D, single is [1,H,W] so squeeze channel dim
            img = single.squeeze(0)  # [H,W]
            F = torch.fft.fft2(img)   # complex [H,W]

            # 강조할 band mask 생성 (flatten 기준)
            flat_size = H * W
            if bands_to_scale:
                # 유효한 인덱스만 필터
                valid_idxs = [i for i in bands_to_scale if 0 <= i < flat_size]
                invalid = [i for i in bands_to_scale if not (0 <= i < flat_size)]
                if invalid:
                    # 한 번만 경고
                    print(f"[full_fft_ifft] 무시된 잘못된 band index: {invalid} (이미지 크기 {H}x{W})")
                if valid_idxs:
                    # 2D 위치로 변환
                    idx_tensor = torch.tensor(valid_idxs, device=F.device)
                    rows = idx_tensor // W
                    cols = idx_tensor % W
                    # 스케일 적용
                    F[rows, cols] = F[rows, cols] * scale_factor

            # IFFT2 → 실수부
            recon = torch.fft.ifft2(F).real  # [H,W]

            # 원래 범위 벗어나는 값만 clamp
            recon = recon.clamp(orig_min.item(), orig_max.item())

            out[b, c] = recon.unsqueeze(0)  # [1,H,W] assign

    if not is_batch:
        out = out.squeeze(0)  # [C,H,W]
    return out