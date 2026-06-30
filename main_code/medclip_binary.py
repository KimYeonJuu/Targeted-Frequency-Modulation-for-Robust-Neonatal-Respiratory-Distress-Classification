import torch
import torch.nn as nn
import types
from typing import Dict
from medclip import MedCLIPVisionModelViT, MedCLIPVisionModel
from main_code.SpectralGatingNetwork import Block as SpectralBlock
from main_code.SpectralGatingNetwork import SpectralGatingNetwork

def _normalize_fb_mode(mode) -> str:
    if isinstance(mode, str):
        return "ON" if mode.upper() == "ON" else "OFF"
    return "ON" if bool(mode) else "OFF"


# ===============================
# Shared utility: parse and apply freeze mode.
# ===============================
def _apply_freeze_policy(backbone: nn.Module, mode: str):
    """
    mode == "OFF": train the full backbone.
    mode == "ON": freeze the backbone and train only spectral modules.
    """
    mode = mode.upper() if isinstance(mode, str) else ("ON" if bool(mode) else "OFF")

    if mode == "OFF":
        for p in backbone.parameters():
            p.requires_grad = True
        return

    # 1) Freeze the full backbone.
    for p in backbone.parameters():
        p.requires_grad = False

    # 2) Re-enable only spectral modules, shared by replacement and insertion modes.
    for m in backbone.modules():
        if isinstance(m, (SpectralBlock, SpectralGatingNetwork, HFSwinBlockAdapter)):
            for p in m.parameters():
                p.requires_grad = True


# ===============================
# Basic binary classifier (ViT / ResNet).
# ===============================
class MedCLIPBinaryClassifier(nn.Module):
    def __init__(self, backbone: str = "vit", num_classes: int = 2, dropout: float = 0.0, freeze_backbone="OFF"):
        super().__init__()
        bb = backbone.lower()
        if bb == "vit":
            self.backbone = MedCLIPVisionModelViT()
            embed_dim = self.backbone.model.config.hidden_size
            self.backbone_type = "vit"
        elif bb == "resnet":
            self.backbone = MedCLIPVisionModel()
            embed_dim = 512  # projection-head output
            self.backbone_type = "resnet"
        else:
            raise ValueError("backbone must be 'vit' or 'resnet'.")

        # Apply the freeze policy. ViT/ResNet do not include SGN, so ON freezes the full backbone.
        fb_mode = _normalize_fb_mode(freeze_backbone)
        if self.backbone_type in ("vit", "resnet"):
            if fb_mode == "ON":
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "vit":
            feats = self.backbone(pixel_values=x, project=False)  # (B, hidden_size)
        else:  # resnet
            feats = self.backbone(pixel_values=x)  # (B, 512)
        return self.head(self.dropout(feats))


# ==========================================
# Spectral Block adapter, compatible with HF Swin.
# ==========================================
class HFSwinBlockAdapter(nn.Module):
    """
    - Input: hidden_states (B, N, C), input_dimensions=(H, W)
    - Output: (hidden_states,) or (hidden_states, None) when output_attentions is enabled
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sb_id=None):
        super().__init__()
        self.block = SpectralBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, sb_id=sb_id)

    def forward(self, hidden_states: torch.Tensor, input_dimensions, attention_mask=None, head_mask=None, output_attentions: bool = False):
        H, W = input_dimensions
        out = self.block(hidden_states, H, W)
        return (out, None) if output_attentions else (out,)


# ==================================================
# Replacement mode: replace Swin blocks specified as "stage:block".
# ==================================================
def parse_block_selection(sb_spec: str, depths) -> set:
    """
    Examples using 1-based indices:
      "1:1,1:2;2:1"  -> S1 B1,B2 / S2 B1
      "1:*;2:1-2"    -> all S1 blocks / S2 B1..2
      "3:4,3:6,4:2"  -> S3 B4,B6 / S4 B2
      "2"            -> all S2 blocks
    Return: set[(stage0, block0)].
    """
    if not sb_spec or not sb_spec.strip():
        return set()
    spec = sb_spec.replace(";", ",").replace(" ", "")
    items = [tok for tok in spec.split(",") if tok]
    sel, num_stages = set(), len(depths)

    for it in items:
        if ":" in it:
            s_txt, b_txt = it.split(":", 1)
        else:
            s_txt, b_txt = it, "*"

        if not s_txt.isdigit():
            raise ValueError(f"Invalid stage specification: '{it}' (1..{num_stages})")
        s1 = int(s_txt)
        if not (1 <= s1 <= num_stages):
            raise ValueError(f"stage out of range: {s1} (1..{num_stages})")
        s0 = s1 - 1
        depth = depths[s0]

        def add_block(b1):
            if not (1 <= b1 <= depth):
                raise ValueError(f"stage {s1} block out of range: {b1} (1..{depth})")
            sel.add((s0, b1 - 1))

        if b_txt in ("*", ""):
            for b in range(1, depth + 1):
                add_block(b)
        elif "-" in b_txt:
            a, b = b_txt.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                raise ValueError(f"Invalid range specification: '{it}'")
            a1, b1 = int(a), int(b)
            if a1 > b1:
                a1, b1 = b1, a1
            for k in range(a1, b1 + 1):
                add_block(k)
        else:
            if not b_txt.isdigit():
                raise ValueError(f"Invalid block specification: '{it}'")
            add_block(int(b_txt))
    return sel


def apply_spectral_blocks_swin(stages, *, config, sb_spec: str):
    """
    stages: hf_model.encoder.layers (list[SwinStage])
    sb_spec: "1:1,2:1" style replacement target specification
    """
    depths = list(getattr(config, "depths"))
    selections = parse_block_selection(sb_spec, depths)
    if not selections:
        print("[SB-REPLACE] no specification provided; keeping the original Swin.")
        return

    for s0, b0 in sorted(selections):
        stage = stages[s0]
        depth = len(stage.blocks)
        if not (0 <= b0 < depth):
            raise ValueError(f"[SB-REPLACE] stage {s0+1} block {b0+1} out of range (1..{depth})")
        dim = config.embed_dim * (2 ** s0)
        heads = config.num_heads[s0] if isinstance(config.num_heads, (list, tuple)) else config.num_heads
        mlp_r = getattr(config, "mlp_ratio", 4.0)
        stage.blocks[b0] = HFSwinBlockAdapter(dim=dim, num_heads=heads, mlp_ratio=mlp_r, sb_id=f"S{s0+1}B{b0+1}")
        print(f"[SB-REPLACE] stage {s0+1} block {b0+1} -> replaced with Spectral Block")


class MedCLIPVisionModelViT_SGN(MedCLIPVisionModelViT):
    """Swin block replacement mode."""
    def __init__(self, checkpoint=None, medclip_checkpoint=None, sb_spec: str = ""):
        super().__init__(checkpoint=checkpoint, medclip_checkpoint=medclip_checkpoint)
        hf_model = self.model
        if not (hasattr(hf_model, "encoder") and hasattr(hf_model.encoder, "layers")):
            raise NotImplementedError("Only Swin backbones are supported.")
        cfg, stages = hf_model.config, hf_model.encoder.layers
        apply_spectral_blocks_swin(stages, config=cfg, sb_spec=sb_spec)
        print(f"[SGN-Swin REPLACE] '{sb_spec}' applied (embed_dim={cfg.embed_dim}, depths={cfg.depths})")


class MedCLIPBinaryClassifier_SGN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.0, freeze_backbone="OFF", sb_spec: str = ""):
        super().__init__()
        self.backbone = MedCLIPVisionModelViT_SGN(sb_spec=sb_spec)
        cfg = self.backbone.model.config
        embed_dim = getattr(cfg, "hidden_size", cfg.embed_dim * (2 ** (len(cfg.depths) - 1)))
        _apply_freeze_policy(self.backbone, freeze_backbone)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values=x, project=False)
        return self.head(self.dropout(feats))


# ==================================================
# Insertion mode: insert immediately after the stage input module, LinearEmb or PatchMerging.
# ==================================================
def parse_sb_insert_spec(spec: str, num_stages: int) -> Dict[int, int]:
    """
    Examples: "1:1" inserts one block in S1, "1:1-2" inserts two in S1, "1:2,3:1".
    Return: dict[stage0] = count.
    """
    if not spec or not spec.strip():
        return {}
    spec = spec.replace(";", ",").replace(" ", "")
    out = {}
    for tok in [t for t in spec.split(",") if t]:
        if ":" not in tok:
            raise ValueError(f"'{tok}': must follow the 'stage:count' format.")
        s_txt, c_txt = tok.split(":", 1)
        if not s_txt.isdigit():
            raise ValueError(f"Invalid stage specification: {s_txt}")
        s1 = int(s_txt)
        if not (1 <= s1 <= num_stages):
            raise ValueError(f"stage out of range: {s1} (1..{num_stages})")
        s0 = s1 - 1

        if "-" in c_txt:
            a, b = c_txt.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                raise ValueError(f"Invalid count range specification: {c_txt}")
            a1, b1 = int(a), int(b)
            if a1 > b1:
                a1, b1 = b1, a1
            count = b1 - a1 + 1
        else:
            if not c_txt.isdigit():
                raise ValueError(f"Invalid count specification: {c_txt}")
            count = int(c_txt)
            if count <= 0:
                raise ValueError("count must be at least 1.")

        out[s0] = out.get(s0, 0) + count
    return out


class _IdentityDownsample(nn.Module):
    def forward(self, hidden_states, input_dimensions):
        return hidden_states, input_dimensions


class _DownsampleThenPreSB(nn.Module):
    def __init__(self, downsample: nn.Module, pre_sbs: nn.ModuleList):
        super().__init__()
        self.downsample = downsample
        self.pre_sbs = pre_sbs  # ModuleList of HFSwinBlockAdapter

    def forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False):
        # 1) Call the original downsample; return type may be tensor or tuple depending on version.
        ds_out = self.downsample(hidden_states, input_dimensions)

        # 2) Extract hidden_states and (H,W), required by the spectral block.
        if isinstance(ds_out, tuple):
            hs = ds_out[0]
            # Keep resolution for stage 1 identity; otherwise use the updated tuple resolution when available.
            if isinstance(self.downsample, _IdentityDownsample):
                new_hw = input_dimensions
            else:
                if len(ds_out) >= 2 and isinstance(ds_out[1], tuple):
                    new_hw = ds_out[1]
                else:
                    # Fallback: assume patch merging halves the resolution, rounding up for padding.
                    H, W = input_dimensions
                    new_hw = ((H + 1) // 2, (W + 1) // 2)
        else:
            # Version returning only a tensor.
            hs = ds_out
            if isinstance(self.downsample, _IdentityDownsample):
                new_hw = input_dimensions  # stage 1 keeps the same resolution.
            else:
                H, W = input_dimensions  # Assume patch merging halves the resolution.
                new_hw = ((H + 1) // 2, (W + 1) // 2)

        # 3) Apply spectral blocks after Patch Merging or Identity.
        for sb in self.pre_sbs:
            hs = sb(hs, new_hw, None, None, False)[0]

        # 4) Preserve the original downsample return format.
        if isinstance(ds_out, tuple):
            out_list = list(ds_out)
            out_list[0] = hs
            # If the version returns resolution in the tuple, insert the updated new_hw.
            if len(out_list) >= 2 and isinstance(out_list[1], tuple):
                out_list[1] = new_hw
            return tuple(out_list)
        else:
            return hs


def insert_spectral_before_stage_blocks(stages, *, config, sb_insert_spec: str):
    """
    Insert k spectral blocks before the blocks in each stage:
      S1: Linear Embedding -> SB x k -> SwinBlock x 2
      S2: Patch Merging -> SB x k -> SwinBlock x 2
      S3: Patch Merging -> SB x k -> SwinBlock x 6
      S4: Patch Merging -> SB x k -> SwinBlock x 2
    """
    depths = list(getattr(config, "depths"))
    plan = parse_sb_insert_spec(sb_insert_spec, len(depths))
    if not plan:
        print("[SB-INSERT] no specification provided; keeping the original Swin.")
        return

    for s0, count in sorted(plan.items()):
        stage = stages[s0]
        # Input channels for each stage, i.e. the dim received by the blocks.
        dim = config.embed_dim * (2 ** s0)
        heads = config.num_heads[s0] if isinstance(config.num_heads, (list, tuple)) else config.num_heads
        mlp_r = getattr(config, "mlp_ratio", 4.0)

        # Spectral blocks applied first at the stage entrance.
        pre_sbs = nn.ModuleList([
            HFSwinBlockAdapter(
                dim=dim, num_heads=heads, mlp_ratio=mlp_r, sb_id=f"S{s0+1}I{k+1}"
            ) for k in range(count)
        ])
        # Store on the stage object.
        stage._pre_sbs = pre_sbs

        # Backup the original forward once.
        if not hasattr(stage, "_orig_forward"):
            stage._orig_forward = stage.forward

        # New forward: spectral blocks followed by the original stage.forward.
        def new_forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False):
            # Apply spectral blocks first without head_mask.
            for sb in self._pre_sbs:
                hidden_states = sb(hidden_states, input_dimensions, None, None, False)[0]
            # Then run the original stage logic, including block loops and downsampling when needed.
            return self._orig_forward(hidden_states, input_dimensions, head_mask, output_attentions)

        # Bind the method.
        stage.forward = types.MethodType(new_forward, stage)
        print(f"[SB-INSERT] Stage {s0+1}: insert SB x{count} before stage blocks (dim={dim})")


class MedCLIPVisionModelViT_SGN_Inserted(MedCLIPVisionModelViT):
    """Swin stage-entry insertion mode."""
    def __init__(
        self, checkpoint=None, medclip_checkpoint=None,
        # Accept both arguments, preferring sb_insert_spec.
        sb_insert_spec: str = "",
        sb_spec: str = ""
    ):
        super().__init__(checkpoint=checkpoint, medclip_checkpoint=medclip_checkpoint)
        hf_model = self.model
        if not (hasattr(hf_model, "encoder") and hasattr(hf_model.encoder, "layers")):
            raise NotImplementedError("Only Swin backbones are supported.")
        cfg, stages = hf_model.config, hf_model.encoder.layers
        plan_spec = sb_insert_spec if sb_insert_spec else sb_spec  # Compatibility alias.
        insert_spectral_before_stage_blocks(stages, config=cfg, sb_insert_spec=plan_spec)
        print(f"[SGN-Swin INSERT] '{plan_spec}' applied (embed_dim={cfg.embed_dim}, depths={cfg.depths})")


class MedCLIPBinaryClassifier_SGN_Inserted(nn.Module):
    def __init__(
        self, num_classes: int = 2, dropout: float = 0.0, freeze_backbone="OFF",
        # Accept both arguments, preferring sb_insert_spec.
        sb_insert_spec: str = "",
        sb_spec: str = ""
    ):
        super().__init__()
        plan_spec = sb_insert_spec if sb_insert_spec else sb_spec  # Compatibility alias.
        self.backbone = MedCLIPVisionModelViT_SGN_Inserted(sb_insert_spec=plan_spec)
        cfg = self.backbone.model.config
        embed_dim = getattr(cfg, "hidden_size", cfg.embed_dim * (2 ** (len(cfg.depths) - 1)))
        _apply_freeze_policy(self.backbone, freeze_backbone)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values=x, project=False)
        return self.head(self.dropout(feats))
