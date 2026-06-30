import os
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class ChannelSELayer(nn.Module):
    """Channel squeeze-and-excitation layer used by the spectral gate."""

    def __init__(self, num_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(1, num_channels // reduction_ratio)
        self.fc1 = nn.Linear(num_channels, reduced_channels, bias=True)
        self.fc2 = nn.Linear(reduced_channels, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _, _ = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        fc_out = self.relu(self.fc1(squeeze_tensor))
        fc_out = self.sigmoid(self.fc2(fc_out))
        return input_tensor * fc_out.view(batch_size, num_channels, 1, 1)


# ===== SGN global defaults =====
_SGN_DEFAULTS = dict(
    mode="grid",              # "grid" | "block"
    grid_size=(16, 16),       # Grid-parameter resolution kept for compatibility.
    tau=0.25,                 # Gating temperature kept for compatibility.
    amp=2.0,                  # Kept for compatibility; not used for spatial scaling.
    block_size=(16, 16),      # Block-mode kernel.
    block_stride=None,        # None means non-overlapping blocks with stride = block_size.

    # Outlier-selection options used to construct the binary mask.
    use_outlier_mask=True,
    ratio_thresh=0.02,
    outlier_topk=0,
    min_radius=1,
    max_radius=-1,

    # Input-derived mask threshold, used only when no external lung mask is provided.
    nz_threshold=1.0 / 255.0,

    # Channel SE reduction ratio.
    se_reduction=16,

    # Maximum channel-wise correction magnitude.
    delta_max=1.0,
)


def set_sgn_defaults(**kwargs):
    _SGN_DEFAULTS.update(kwargs)


class SpectralGatingNetwork(nn.Module):
    """
    Spectral gating network with outlier-band masking and SE-based channel scaling.

    The module first estimates a sample-wise outlier-frequency mask from rFFT
    magnitudes. It then computes SE-derived channel scores from the selected
    spectral bins and applies the resulting channel-wise scale only inside the
    selected outlier region. Unselected bins pass through unchanged.
    """

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

        # Legacy grid parameters retained for checkpoint/config compatibility.
        self.Gh, self.Gw = cfg.get("grid_size", (16, 16))
        self.grid_logits = nn.Parameter(torch.zeros(self.Gh, self.Gw))
        nn.init.normal_(self.grid_logits, mean=0.0, std=0.02)

        # Block-mode parameters.
        self.block_h, self.block_w = cfg["block_size"]
        self.block_stride = cfg["block_stride"] or (self.block_h, self.block_w)

        # Outlier-selection options.
        self.use_outlier_mask = bool(cfg.get("use_outlier_mask", True))
        self.ratio_thresh = float(cfg.get("ratio_thresh", 0.02))
        self.outlier_topk = int(cfg.get("outlier_topk", 0))
        self.min_radius = int(cfg.get("min_radius", 1))
        self.max_radius = int(cfg.get("max_radius", -1))

        # Threshold used when no external mask is provided.
        self.nz_threshold = float(cfg.get("nz_threshold", 1.0 / 255.0))

        # Channel SE module and learnable channel-scaling parameters.
        se_reduction = int(cfg.get("se_reduction", 16))
        self.cse = ChannelSELayer(num_channels=dim, reduction_ratio=se_reduction)
        self.log_gamma = nn.Parameter(torch.full((dim,), math.log(2.0), dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.delta_max = float(cfg.get("delta_max", 1.0))

    # ----------------- internal utilities -----------------
    @staticmethod
    def _resize_mask(mask: torch.Tensor, H: int, W: int, ch_last: bool):
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        mask = (mask > 0.5).to(dtype=torch.float32)
        return mask.permute(0, 2, 3, 1).contiguous() if ch_last else mask

    def _pop_ext_mask(self):
        mask = getattr(self, "_ext_mask", None)
        if mask is not None:
            delattr(self, "_ext_mask")
        return mask

    def _apply_outlier_mask(self, Xf: torch.Tensor, scale_2d: torch.Tensor):
        """
        Build the outlier-frequency mask.

        Returns a per-sample 2D tensor where selected outlier positions receive
        ``scale_2d`` and all other positions receive 1.0. In the SE path, this is
        used only to recover the binary selected-bin mask.
        """
        B, Hf, Wf, _ = Xf.shape
        device = Xf.device

        if not self.use_outlier_mask or (self.ratio_thresh <= 0.0 and self.outlier_topk <= 0):
            return scale_2d.unsqueeze(0).expand(B, Hf, Wf)

        mag = torch.log1p(torch.abs(Xf)).mean(dim=-1)  # (B,Hf,Wf)
        den = mag.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        ratio = mag / den

        iy = torch.arange(Hf, device=device, dtype=torch.float32).view(Hf, 1)
        ix = torch.arange(Wf, device=device, dtype=torch.float32).view(1, Wf)
        radius = torch.sqrt(iy * iy + ix * ix)
        mask_radius = radius >= float(self.min_radius)
        if self.max_radius >= 0:
            mask_radius = mask_radius & (radius <= float(self.max_radius))

        if self.ratio_thresh > 0.0:
            candidate_mask = (ratio >= self.ratio_thresh) & mask_radius
        else:
            candidate_mask = torch.zeros_like(ratio, dtype=torch.bool)

        if self.outlier_topk > 0:
            pool = mask_radius.unsqueeze(0).expand(B, Hf, Wf)
            flat_ratio = ratio.flatten(1).clone().masked_fill(~pool.flatten(1), float("-inf"))
            k = min(self.outlier_topk, flat_ratio.size(1))
            _, topk_idx = torch.topk(flat_ratio, k=k, dim=1)
            topk_mask = torch.zeros_like(flat_ratio, dtype=torch.bool)
            topk_mask.scatter_(1, topk_idx, True)
            topk_mask = topk_mask.view(B, Hf, Wf)
            union = candidate_mask | topk_mask

            union_flat = union.flatten(1)
            flat_ratio_union = ratio.flatten(1).clone().masked_fill(~union_flat, float("-inf"))
            _, topk_idx_union = torch.topk(flat_ratio_union, k=k, dim=1)
            final_mask = torch.zeros_like(union_flat, dtype=torch.bool)
            final_mask.scatter_(1, topk_idx_union, True)
            final_mask = final_mask.view(B, Hf, Wf)
        else:
            final_mask = candidate_mask

        # Rescue rule: ensure at least one selected coordinate for each sample.
        empty = (~final_mask).all(dim=(1, 2))
        if empty.any():
            pool = mask_radius.unsqueeze(0).expand(B, Hf, Wf)
            flat_ratio = ratio.flatten(1).clone().masked_fill(~pool.flatten(1), float("-inf"))
            _, top1_idx = torch.topk(flat_ratio, k=1, dim=1)
            rescue = torch.zeros_like(flat_ratio, dtype=torch.bool)
            rescue.scatter_(1, top1_idx, True)
            rescue = rescue.view(B, Hf, Wf)
            final_mask[empty] = rescue[empty]

        base = scale_2d.unsqueeze(0).expand(B, Hf, Wf)
        final_mask = final_mask.to(torch.bool)
        return torch.where(final_mask, base, torch.ones_like(base))

    def _se_gate(self, Xf: torch.Tensor, mask2d: Optional[torch.Tensor] = None):
        """
        Compute per-sample, per-channel SE scores from spectral magnitudes.

        Parameters
        ----------
        Xf:
            Complex spectrum with shape ``(B,Hf,Wf,C)``.
        mask2d:
            Optional binary outlier mask with shape ``(B,Hf,Wf)``. When provided,
            pooling is restricted to the selected outlier bins.
        """
        _, _, _, C = Xf.shape
        mag = torch.abs(Xf)

        if mask2d is not None:
            mask = mask2d[..., None].to(mag.dtype)
            numerator = (mag * mask).sum(dim=(1, 2))
            denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
            pooled = numerator / denominator
        else:
            pooled = mag.mean(dim=(1, 2))

        # Reuse the ChannelSELayer MLP weights directly because its forward
        # method expects a 4D feature map.
        z = self.cse.fc2(F.relu(self.cse.fc1(pooled)))
        return torch.sigmoid(z).view(-1, C)

    # ----------------- processing paths -----------------
    def _grid_process(self, x, H, W):
        B, N, C = x.shape
        x_img = x.view(B, H, W, C).to(torch.float32)

        # Apply an external lung mask when provided; otherwise derive a simple
        # foreground mask from the input feature map.
        ext = self._pop_ext_mask()
        if ext is not None:
            hard = self._resize_mask(ext, H, W, ch_last=True)
        else:
            gray = x_img.mean(dim=-1, keepdim=True)
            mn = gray.amin(dim=(1, 2, 3), keepdim=True)
            mx = gray.amax(dim=(1, 2, 3), keepdim=True)
            rng = (mx - mn).clamp_min(1e-8)
            gray_norm = (gray - mn) / rng
            hard = (gray_norm > self.nz_threshold).to(x_img.dtype)

        x_in = x_img * hard

        Xf = torch.fft.rfft2(x_in, dim=(1, 2), norm="ortho")
        Hf, Wf = Xf.shape[1], Xf.shape[2]

        # Build the selected-bin mask. The values 2.0 and 1.0 are used only to
        # recover the boolean outlier mask after selection.
        const2 = torch.full((Hf, Wf), 2.0, device=Xf.device, dtype=torch.float32)
        scale_mask2 = self._apply_outlier_mask(Xf, const2)
        mask2d_bool = scale_mask2 > 1.0
        mask2d = mask2d_bool.to(Xf.real.dtype)

        # Compute channel-wise scaling only for selected bins.
        gamma = torch.exp(self.log_gamma)[None, :]
        gate = self._se_gate(Xf, mask2d=mask2d_bool)
        alpha = 0.3
        delta = torch.tanh(self.beta)[None, :] * self.delta_max * (2.0 * gate - 1.0) * alpha

        scale_ch = (gamma - 1.0) + delta
        final_scale = 1.0 + mask2d[..., None] * scale_ch[:, None, None, :]
        final_scale = final_scale.clamp_min(0.0)

        Xf = Xf * final_scale

        x_rec = torch.fft.irfft2(Xf, s=(H, W), dim=(1, 2), norm="ortho").to(x.dtype)
        x_rec = x_rec * hard
        return x_rec.view(B, N, C)

    def _block_process(self, x, H, W):
        B, N, C = x.shape
        x_chw = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous().to(torch.float32)

        # Apply an external lung mask when provided; otherwise derive a simple
        # foreground mask from the input feature map.
        ext = self._pop_ext_mask()
        if ext is not None:
            hard = self._resize_mask(ext, H, W, ch_last=False)
        else:
            gray = x_chw.mean(dim=1, keepdim=True)
            mn = gray.amin(dim=(2, 3), keepdim=True)
            mx = gray.amax(dim=(2, 3), keepdim=True)
            rng = (mx - mn).clamp_min(1e-8)
            gray_norm = (gray - mn) / rng
            hard = (gray_norm > self.nz_threshold).to(x_chw.dtype)

        x_chw_in = x_chw * hard

        unfold = nn.Unfold(kernel_size=(self.block_h, self.block_w), stride=self.block_stride)
        fold = nn.Fold(output_size=(H, W), kernel_size=(self.block_h, self.block_w), stride=self.block_stride)

        patches = unfold(x_chw_in)
        Cbhbw, L = patches.shape[1], patches.shape[2]
        bh, bw = self.block_h, self.block_w
        assert Cbhbw == C * bh * bw, "Unfold shape mismatch"

        patches = patches.transpose(1, 2).contiguous().view(B * L, C, bh, bw)
        patches_cl = patches.permute(0, 2, 3, 1).contiguous()

        Xf = torch.fft.rfft2(patches_cl, dim=(1, 2), norm="ortho")
        bhf, bwf = Xf.shape[1], Xf.shape[2]

        # Build the selected-bin mask. The values 2.0 and 1.0 are used only to
        # recover the boolean outlier mask after selection.
        const2 = torch.full((bhf, bwf), 2.0, device=Xf.device, dtype=torch.float32)
        scale_mask2 = self._apply_outlier_mask(Xf, const2)
        mask2d_bool = scale_mask2 > 1.0
        mask2d = mask2d_bool.to(Xf.real.dtype)

        # Compute channel-wise scaling only for selected bins.
        gamma = torch.exp(self.log_gamma)[None, :]
        gate = self._se_gate(Xf, mask2d=mask2d_bool)
        alpha = 0.3
        delta = torch.tanh(self.beta)[None, :] * self.delta_max * (2.0 * gate - 1.0) * alpha

        scale_ch = (gamma - 1.0) + delta
        final_scale = 1.0 + mask2d[..., None] * scale_ch[:, None, None, :]
        final_scale = final_scale.clamp_min(0.0)

        Xf = Xf * final_scale

        x_rec = torch.fft.irfft2(Xf, s=(bh, bw), dim=(1, 2), norm="ortho")
        x_rec = x_rec.permute(0, 3, 1, 2).contiguous()

        x_rec = x_rec.view(B, L, C * bh * bw).transpose(1, 2).contiguous()
        x_chw_rec = fold(x_rec)

        x_chw_rec = x_chw_rec * hard
        return x_chw_rec.permute(0, 2, 3, 1).contiguous().to(x.dtype).view(B, N, C)

    # ----------------- interface -----------------
    def set_external_mask(self, mask: torch.Tensor):
        """
        Set an external binary mask for the next forward pass only.

        ``mask`` may have shape ``(B,1,H0,W0)`` or ``(B,H0,W0)``. Values greater
        than zero are treated as foreground.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0).to(dtype=torch.float32)
        self._ext_mask = mask.detach().contiguous()

    def forward(self, x: torch.Tensor, H: int, W: int):
        if self.mode == "grid":
            return self._grid_process(x, H, W)
        if self.mode == "block":
            return self._block_process(x, H, W)
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
        self.attn = SpectralGatingNetwork(dim, sb_id=sb_id)
        self.norm2 = norm_layer(dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
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
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
