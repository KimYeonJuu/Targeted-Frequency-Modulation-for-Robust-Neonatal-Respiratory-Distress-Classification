import os
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


# ===== SGN global defaults =====
_SGN_DEFAULTS = dict(
    mode="grid",              # "grid" | "block"
    grid_size=(16, 16),       # Grid size, i.e. gate-parameter resolution.
    tau=0.25,                 # Gating temperature.
    amp=2.0,                  # scale = 1 + amp * gate
    block_size=(16, 16),      # Block-mode kernel.
    block_stride=None,        # None -> same as block_size, non-overlapping.

    # Outlier-selection options.
    use_outlier_mask=True,    # If True, scale only outlier bands.
    ratio_thresh=0.02,        # Energy-ratio threshold.
    outlier_topk=0,           # Top K; 0 disables this option.
    min_radius=1,             # Exclude low frequencies using index-coordinate radius.
    max_radius=-1,            # Maximum radius; negative disables this option.

    # Input-derived mask threshold, used only when no external mask is provided.
    nz_threshold=1.0 / 255.0,
)

def set_sgn_defaults(**kwargs):
    _SGN_DEFAULTS.update(kwargs)


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim: int, sb_id: Optional[Union[int, str]] = None, **kwargs):
        super().__init__()
        cfg = dict(_SGN_DEFAULTS)
        env_mode = os.getenv("SGN_MODE", "").strip().lower()
        if env_mode in ("grid", "block"):
            cfg["mode"] = env_mode
        cfg.update(kwargs)

        self.sb_id = sb_id
        self.dim = dim
        self.mode = cfg["mode"]
        self.tau = float(cfg["tau"])
        self.amp = float(cfg["amp"])

        # Grid gate parameters: bilinearly resize (Gh,Gw) to (Hf,Wf).
        self.Gh, self.Gw = cfg.get("grid_size", (16, 16))
        self.grid_logits = nn.Parameter(torch.zeros(self.Gh, self.Gw))
        nn.init.normal_(self.grid_logits, mean=0.0, std=0.02)

        # Block-mode parameters.
        self.block_h, self.block_w = cfg["block_size"]
        self.block_stride = cfg["block_stride"] or (self.block_h, self.block_w)

        # Outlier-selection options.
        self.use_outlier_mask = bool(cfg.get("use_outlier_mask", True))
        self.ratio_thresh     = float(cfg.get("ratio_thresh", 0.02))
        self.outlier_topk     = int(cfg.get("outlier_topk", 0))
        self.min_radius       = int(cfg.get("min_radius", 1))
        self.max_radius       = int(cfg.get("max_radius", -1))

        # Threshold used when no external mask is provided.
        self.nz_threshold = float(kwargs.get("nz_threshold", 1.0 / 255.0))

    # ----------------- internal utilities -----------------
    @staticmethod
    def _resize_mask(mask: torch.Tensor, H: int, W: int, ch_last: bool):
        """mask: (B,1,h,w) -> nearest resize -> (B,1,H,W) or (B,H,W,1)."""
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        mask = (mask > 0.5).to(dtype=torch.float32)
        return mask.permute(0, 2, 3, 1).contiguous() if ch_last else mask

    def _pop_ext_mask(self):
        m = getattr(self, "_ext_mask", None)
        if m is not None:
            delattr(self, "_ext_mask")  # Use only once.
        return m

    def _scale_map_base(self, Hf: int, Wf: int, device):
        g = F.interpolate(
            self.grid_logits.unsqueeze(0).unsqueeze(0),  # (1,1,Gh,Gw)
            size=(Hf, Wf),
            mode="bilinear",
            align_corners=False,
        )[0, 0]  # (Hf,Wf)
        gate  = torch.sigmoid(g / (self.tau + 1e-8))
        scale = 1.0 + self.amp * gate
        return scale.to(device)

    def _apply_outlier_mask(self, Xf: torch.Tensor, scale_2d: torch.Tensor):
        """
        Xf: (B, Hf, Wf, C) complex
        scale_2d: (Hf, Wf) real
        Return: (B,Hf,Wf) per-batch scale with mask applied.
        """
        B, Hf, Wf, C = Xf.shape
        device = Xf.device

        if not self.use_outlier_mask or (self.ratio_thresh <= 0.0 and self.outlier_topk <= 0):
            return scale_2d.expand(B, Hf, Wf)

        mag = torch.log1p(torch.abs(Xf)).mean(dim=-1)  # (B,Hf,Wf)
        den = mag.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        r = mag / den                                   # (B,Hf,Wf)

        iy = torch.arange(Hf, device=device, dtype=torch.float32).view(Hf, 1)
        ix = torch.arange(Wf, device=device, dtype=torch.float32).view(1, Wf)
        rad = torch.sqrt(iy * iy + ix * ix)
        mask_radius = (rad >= float(self.min_radius))
        if self.max_radius >= 0:
            mask_radius = mask_radius & (rad <= float(self.max_radius))

        cand_thr = (r >= self.ratio_thresh) & mask_radius if self.ratio_thresh > 0.0 \
            else torch.zeros_like(r, dtype=torch.bool)

        if self.outlier_topk > 0:
            pool = mask_radius.unsqueeze(0).expand(B, Hf, Wf)
            flat_r = r.flatten(1).clone().masked_fill(~pool.flatten(1), float("-inf"))
            k = min(self.outlier_topk, flat_r.size(1))
            _, topk_idx = torch.topk(flat_r, k=k, dim=1)
            topk_mask = torch.zeros_like(flat_r, dtype=torch.bool)
            topk_mask.scatter_(1, topk_idx, True)
            topk_mask = topk_mask.view(B, Hf, Wf)
            union = cand_thr | topk_mask

            union_flat = union.flatten(1)
            flat_r2 = r.flatten(1).clone().masked_fill(~union_flat, float("-inf"))
            _, topk_idx2 = torch.topk(flat_r2, k=k, dim=1)
            final_mask = torch.zeros_like(union_flat, dtype=torch.bool)
            final_mask.scatter_(1, topk_idx2, True)
            final_mask = final_mask.view(B, Hf, Wf)
        else:
            final_mask = cand_thr

        empty = (~final_mask).all(dim=(1, 2))
        if empty.any():
            pool = mask_radius.unsqueeze(0).expand(B, Hf, Wf)
            flat_r = r.flatten(1).clone().masked_fill(~pool.flatten(1), float("-inf"))
            _, top1_idx = torch.topk(flat_r, k=1, dim=1)
            rescue = torch.zeros_like(flat_r, dtype=torch.bool)
            rescue.scatter_(1, top1_idx, True)
            rescue = rescue.view(B, Hf, Wf)
            final_mask[empty] = rescue[empty]

        base = scale_2d.unsqueeze(0).expand(B, -1, -1)  # (B,Hf,Wf)
        out_scale = torch.ones_like(base)
        out_scale[final_mask] = base[final_mask]
        return out_scale

    # ----------------- processing paths -----------------
    def _grid_process(self, x, H, W):
        B, N, C = x.shape
        x_img = x.view(B, H, W, C).to(torch.float32)

        # 1) Use an external mask when available -> (B,H,W,1).
        ext = self._pop_ext_mask()
        if ext is not None:
            hard = self._resize_mask(ext, H, W, ch_last=True)  # nearest-neighbor resize and binarization
        else:
            # 2) Otherwise create a temporary mask from the input.
            g = x_img.mean(dim=-1, keepdim=True)                # (B,H,W,1)
            mn = g.amin(dim=(1, 2, 3), keepdim=True)
            mx = g.amax(dim=(1, 2, 3), keepdim=True)
            rng = (mx - mn).clamp_min(1e-8)
            g_norm = (g - mn) / rng                              # [0,1]
            hard = (g_norm > self.nz_threshold).to(x_img.dtype)  # (B,H,W,1)

        # Mask before FFT.
        x_in = x_img * hard

        # FFT -> gate -> IFFT.
        Xf = torch.fft.rfft2(x_in, dim=(1, 2), norm="ortho")    # (B,Hf,Wf,C)
        Hf, Wf = Xf.shape[1], Xf.shape[2]
        base_scale = self._scale_map_base(Hf, Wf, Xf.device)    # (Hf,Wf)
        scale_b    = self._apply_outlier_mask(Xf, base_scale)   # (B,Hf,Wf)
        Xf = Xf * scale_b[..., None]                            # Scale outlier bands.
        x_rec = torch.fft.irfft2(Xf, s=(H, W), dim=(1, 2), norm="ortho").to(x.dtype)

        # Suppress the background again after IFFT.
        x_rec = x_rec * hard

        return x_rec.view(B, N, C)

    def _block_process(self, x, H, W):
        B, N, C = x.shape
        x_chw = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous().to(torch.float32)  # (B,C,H,W)

        # 1) Use an external mask when available -> (B,1,H,W).
        ext = self._pop_ext_mask()
        if ext is not None:
            hard = self._resize_mask(ext, H, W, ch_last=False)
        else:
            # 2) Otherwise create a temporary mask from the input.
            g = x_chw.mean(dim=1, keepdim=True)                 # (B,1,H,W)
            mn = g.amin(dim=(2, 3), keepdim=True)
            mx = g.amax(dim=(2, 3), keepdim=True)
            rng = (mx - mn).clamp_min(1e-8)
            g_norm = (g - mn) / rng                              # [0,1]
            hard = (g_norm > self.nz_threshold).to(x_chw.dtype)  # (B,1,H,W)

        # Mask before FFT.
        x_chw_in = x_chw * hard

        unfold = nn.Unfold(kernel_size=(self.block_h, self.block_w), stride=self.block_stride)
        fold   = nn.Fold(output_size=(H, W), kernel_size=(self.block_h, self.block_w), stride=self.block_stride)

        patches = unfold(x_chw_in)                                   # (B, C*bh*bw, L)
        Cbhbw, L = patches.shape[1], patches.shape[2]
        bh, bw = self.block_h, self.block_w
        assert Cbhbw == C * bh * bw, "Unfold shape mismatch"

        patches    = patches.transpose(1, 2).contiguous().view(B * L, C, bh, bw)   # (B*L,C,bh,bw)
        patches_cl = patches.permute(0, 2, 3, 1).contiguous()                      # (B*L,bh,bw,C)

        Xf   = torch.fft.rfft2(patches_cl, dim=(1, 2), norm="ortho")               # (B*L,bhf,bwf,C)
        bhf, bwf = Xf.shape[1], Xf.shape[2]
        base_scale = self._scale_map_base(bhf, bwf, Xf.device)                     # (bhf,bwf)
        scale_b    = self._apply_outlier_mask(Xf, base_scale)                      # (B*L,bhf,bwf)
        Xf = Xf * scale_b[..., None]
        x_rec = torch.fft.irfft2(Xf, s=(bh, bw), dim=(1, 2), norm="ortho")         # (B*L,bh,bw,C)
        x_rec = x_rec.permute(0, 3, 1, 2).contiguous()                              # (B*L,C,bh,bw)

        x_rec     = x_rec.view(B, L, C * bh * bw).transpose(1, 2).contiguous()     # (B,C*bh*bw,L)
        x_chw_rec = fold(x_rec)                                                    # (B,C,H,W)

        # Suppress the background again after IFFT.
        x_chw_rec = x_chw_rec * hard

        return x_chw_rec.permute(0, 2, 3, 1).contiguous().to(x.dtype).view(B, N, C)

    # ----------------- interface -----------------
    def set_external_mask(self, mask: torch.Tensor):
        """
        mask: (B,1,H0,W0) or (B,H0,W0); values {0,1} are recommended.
        It is used only for the next forward pass.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B,1,H0,W0)
        mask = (mask > 0).to(dtype=torch.float32)
        self._ext_mask = mask.detach().contiguous()

    def forward(self, x: torch.Tensor, H: int, W: int):
        if self.mode == "grid":
            return self._grid_process(x, H, W)
        elif self.mode == "block":
            return self._block_process(x, H, W)
        else:
            return self._grid_process(x, H, W)


class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        num_heads: Optional[int] = None,
        sb_id=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpectralGatingNetwork(dim, sb_id=sb_id)  # SGN
        self.norm2 = norm_layer(dim)
        self.mlp  = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)
        self.num_heads = num_heads

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # SGN path.
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))   # PVT2-style FFN.
        return x
