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
    Apply block-wise FFT to x, scale selected bands, run IFFT, clamp, and reassemble.
    - x: [C,H,W] or [B,C,H,W]
    - C: number of channels, usually 1 or 3
    - B: batch size
    """
    if bands_to_scale is None:
        bands_to_scale = [8,16,24,32,40,48,56]

    # Convert [C,H,W] input to [1,C,H,W].
    is_batch = (x.dim() == 4)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() != 4:
        raise ValueError(f"Unsupported input dimensions: {x.shape}")

    B, C, H, W = x.shape
    nh, nw = H // block, W // block

    out = torch.zeros_like(x)
    for b in range(B):
        for c in range(C):
            single = x[b, c:c+1]  # shape [1,H,W]

            # Store the original value range.
            orig_min, orig_max = single.min(), single.max()
            # Clamp to the configured range.
            single = single.clamp(clamp_min, clamp_max)

            # Split into blocks.
            blocks = (
                single
                .unfold(1, block, block)
                .unfold(2, block, block)
                .permute(1,2,0,3,4)  # [nh, nw, 1, block, block]
            )[..., 0, :, :]         # [nh, nw, block, block]

            # FFT and scale selected bands.
            B_fft = torch.fft.fft2(blocks)  # complex [nh,nw,block,block]
            B_flat = B_fft.reshape(nh, nw, block*block)
            for idx in bands_to_scale:
                B_flat[:,:,idx] *= scale_factor
            B_fft = B_flat.view(nh, nw, block, block)

            # IFFT and keep the real part.
            B_ifft = torch.fft.ifft2(B_fft).real  # [nh,nw,block,block]

            # Reassemble blocks.
            recon = torch.zeros_like(single)
            for i in range(nh):
                for j in range(nw):
                    recon[0,
                          i*block:(i+1)*block,
                          j*block:(j+1)*block] = B_ifft[i,j]

            # Clamp to the original value range.
            recon = recon.clamp(orig_min.item(), orig_max.item())

            out[b, c] = recon

    # Restore the original dimensionality.
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
    Apply full-image FFT, scale selected frequency indices, run IFFT, and clamp.
    - x: [C,H,W] or [B,C,H,W]
    - bands_to_scale: frequency-bin indices in the flattened FFT (H*W) to scale
    - scale_factor: multiplier applied to the selected bins
    - clamp_min/max: input clamp range, defaulting to 0..1
    Return the same shape as the input, using only the real part.
    """
    if bands_to_scale is None:
        bands_to_scale = []  # No scaling by default.

    # Normalize batch handling.
    is_batch = (x.dim() == 4)
    if x.dim() == 3:
        x = x.unsqueeze(0)  # [1,C,H,W]
    elif x.dim() != 4:
        raise ValueError(f"Unsupported input dimensions: {x.shape}")

    B, C, H, W = x.shape
    out = torch.zeros_like(x)

    for b in range(B):
        for c in range(C):
            single = x[b, c:c+1]  # [1,H,W]

            # Store the original value range.
            orig_min, orig_max = single.min(), single.max()

            # Clamp the input.
            single = single.clamp(clamp_min, clamp_max)

            # Full-image FFT2.
            # torch.fft.fft2 expects at least 2D, single is [1,H,W] so squeeze channel dim
            img = single.squeeze(0)  # [H,W]
            F = torch.fft.fft2(img)   # complex [H,W]

            # Build the band mask in flattened-index coordinates.
            flat_size = H * W
            if bands_to_scale:
                # Keep only valid indices.
                valid_idxs = [i for i in bands_to_scale if 0 <= i < flat_size]
                invalid = [i for i in bands_to_scale if not (0 <= i < flat_size)]
                if invalid:
                    # Warn once.
                    print(f"[full_fft_ifft] Ignoring invalid band index: {invalid} (image size {H}x{W})")
                if valid_idxs:
                    # Convert to 2D positions.
                    idx_tensor = torch.tensor(valid_idxs, device=F.device)
                    rows = idx_tensor // W
                    cols = idx_tensor % W
                    # Apply scaling.
                    F[rows, cols] = F[rows, cols] * scale_factor

            # IFFT2 and keep the real part.
            recon = torch.fft.ifft2(F).real  # [H,W]

            # Clamp to the original value range.
            recon = recon.clamp(orig_min.item(), orig_max.item())

            out[b, c] = recon.unsqueeze(0)  # [1,H,W] assign

    if not is_batch:
        out = out.squeeze(0)  # [C,H,W]
    return out