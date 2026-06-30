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
# 공통 유틸: freeze 모드 해석/적용
# ===============================
def _apply_freeze_policy(backbone: nn.Module, mode: str):
    """
    mode == "OFF": backbone 전체 학습
    mode == "ON" : backbone은 전부 freeze하되 Spectral 계열 모듈만 학습 허용
    """
    mode = mode.upper() if isinstance(mode, str) else ("ON" if bool(mode) else "OFF")

    if mode == "OFF":
        for p in backbone.parameters():
            p.requires_grad = True
        return

    # 1) 전체 동결
    for p in backbone.parameters():
        p.requires_grad = False

    # 2) Spectral 계열만 다시 풀기 (교체/삽입 공통)
    for m in backbone.modules():
        if isinstance(m, (SpectralBlock, SpectralGatingNetwork, HFSwinBlockAdapter)):
            for p in m.parameters():
                p.requires_grad = True


# ===============================
# 기본 이진 분류기 (ViT / ResNet)
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
            embed_dim = 512  # projection head 출력
            self.backbone_type = "resnet"
        else:
            raise ValueError("backbone must be 'vit' or 'resnet'.")

        # freeze 정책 적용 (ViT/ResNet은 SGN이 없으므로 ON이면 전체 freeze)
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
# Spectral Block 어댑터 (HF Swin 호환 래퍼)
# ==========================================
class HFSwinBlockAdapter(nn.Module):
    """
    - 입력: hidden_states (B, N, C), input_dimensions=(H, W)
    - 출력: (hidden_states,) 또는 (hidden_states, None) if output_attentions
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sb_id=None):
        super().__init__()
        self.block = SpectralBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, sb_id=sb_id)

    def forward(self, hidden_states: torch.Tensor, input_dimensions, attention_mask=None, head_mask=None, output_attentions: bool = False):
        H, W = input_dimensions
        out = self.block(hidden_states, H, W)
        return (out, None) if output_attentions else (out,)


# ==================================================
# 교체 모드: "stage:block" 지정으로 Swin 블록 교체
# ==================================================
def parse_block_selection(sb_spec: str, depths) -> set:
    """
    예시(1-기반):
      "1:1,1:2;2:1"  -> S1 B1,B2 / S2 B1
      "1:*;2:1-2"    -> S1 전체 / S2 B1..2
      "3:4,3:6,4:2"  -> S3 B4,B6 / S4 B2
      "2"            -> S2 전체
    반환: set[(stage0, block0)]
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
            raise ValueError(f"잘못된 stage 표기: '{it}' (1..{num_stages})")
        s1 = int(s_txt)
        if not (1 <= s1 <= num_stages):
            raise ValueError(f"stage 범위 오류: {s1} (1..{num_stages})")
        s0 = s1 - 1
        depth = depths[s0]

        def add_block(b1):
            if not (1 <= b1 <= depth):
                raise ValueError(f"stage {s1} block 범위 오류: {b1} (1..{depth})")
            sel.add((s0, b1 - 1))

        if b_txt in ("*", ""):
            for b in range(1, depth + 1):
                add_block(b)
        elif "-" in b_txt:
            a, b = b_txt.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                raise ValueError(f"range 표기 오류: '{it}'")
            a1, b1 = int(a), int(b)
            if a1 > b1:
                a1, b1 = b1, a1
            for k in range(a1, b1 + 1):
                add_block(k)
        else:
            if not b_txt.isdigit():
                raise ValueError(f"block 표기 오류: '{it}'")
            add_block(int(b_txt))
    return sel


def apply_spectral_blocks_swin(stages, *, config, sb_spec: str):
    """
    stages: hf_model.encoder.layers (list[SwinStage])
    sb_spec: "1:1,2:1" 등 (교체 대상 지정)
    """
    depths = list(getattr(config, "depths"))
    selections = parse_block_selection(sb_spec, depths)
    if not selections:
        print("[SB-REPLACE] 지정 없음 (원본 Swin 유지).")
        return

    for s0, b0 in sorted(selections):
        stage = stages[s0]
        depth = len(stage.blocks)
        if not (0 <= b0 < depth):
            raise ValueError(f"[SB-REPLACE] stage {s0+1} block {b0+1} 범위 초과 (1..{depth})")
        dim = config.embed_dim * (2 ** s0)
        heads = config.num_heads[s0] if isinstance(config.num_heads, (list, tuple)) else config.num_heads
        mlp_r = getattr(config, "mlp_ratio", 4.0)
        stage.blocks[b0] = HFSwinBlockAdapter(dim=dim, num_heads=heads, mlp_ratio=mlp_r, sb_id=f"S{s0+1}B{b0+1}")
        print(f"[SB-REPLACE] stage {s0+1} block {b0+1} → Spectral Block 교체 완료")


class MedCLIPVisionModelViT_SGN(MedCLIPVisionModelViT):
    """Swin 블록 교체 모드"""
    def __init__(self, checkpoint=None, medclip_checkpoint=None, sb_spec: str = ""):
        super().__init__(checkpoint=checkpoint, medclip_checkpoint=medclip_checkpoint)
        hf_model = self.model
        if not (hasattr(hf_model, "encoder") and hasattr(hf_model.encoder, "layers")):
            raise NotImplementedError("Swin 백본만 지원합니다.")
        cfg, stages = hf_model.config, hf_model.encoder.layers
        apply_spectral_blocks_swin(stages, config=cfg, sb_spec=sb_spec)
        print(f"[SGN-Swin REPLACE] '{sb_spec}' 적용 (embed_dim={cfg.embed_dim}, depths={cfg.depths})")


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
# 삽입 모드: Stage 시작부(LinearEmb/PatchMerging) 직후 삽입
# ==================================================
def parse_sb_insert_spec(spec: str, num_stages: int) -> Dict[int, int]:
    """
    예) "1:1" (S1에 1개 삽입), "1:1-2"(S1에 2개), "1:2,3:1"
    반환: dict[stage0] = count
    """
    if not spec or not spec.strip():
        return {}
    spec = spec.replace(";", ",").replace(" ", "")
    out = {}
    for tok in [t for t in spec.split(",") if t]:
        if ":" not in tok:
            raise ValueError(f"'{tok}': 'stage:count' 형식이어야 합니다.")
        s_txt, c_txt = tok.split(":", 1)
        if not s_txt.isdigit():
            raise ValueError(f"스테이지 표기 오류: {s_txt}")
        s1 = int(s_txt)
        if not (1 <= s1 <= num_stages):
            raise ValueError(f"stage 범위 오류: {s1} (1..{num_stages})")
        s0 = s1 - 1

        if "-" in c_txt:
            a, b = c_txt.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                raise ValueError(f"count 범위 표기 오류: {c_txt}")
            a1, b1 = int(a), int(b)
            if a1 > b1:
                a1, b1 = b1, a1
            count = b1 - a1 + 1
        else:
            if not c_txt.isdigit():
                raise ValueError(f"count 표기 오류: {c_txt}")
            count = int(c_txt)
            if count <= 0:
                raise ValueError("count는 1 이상이어야 합니다.")

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
        # 1) 원래 downsample 호출: 반환 형식(텐서/튜플)은 버전에 따라 다름
        ds_out = self.downsample(hidden_states, input_dimensions)

        # 2) hidden_states / (H,W) 추출 (SB에 필요)
        if isinstance(ds_out, tuple):
            hs = ds_out[0]
            # stage1(Identity)면 해상도 유지, 그 외엔 가능한 경우 튜플 내에서 갱신된 해상도 사용
            if isinstance(self.downsample, _IdentityDownsample):
                new_hw = input_dimensions
            else:
                if len(ds_out) >= 2 and isinstance(ds_out[1], tuple):
                    new_hw = ds_out[1]
                else:
                    # 안전장치: patch merging 가정 하에 1/2 (패딩 고려해 올림)
                    H, W = input_dimensions
                    new_hw = ((H + 1) // 2, (W + 1) // 2)
        else:
            # 텐서만 반환하는 버전
            hs = ds_out
            if isinstance(self.downsample, _IdentityDownsample):
                new_hw = input_dimensions  # stage1: 해상도 그대로
            else:
                H, W = input_dimensions  # patch merging: 1/2 가정
                new_hw = ((H + 1) // 2, (W + 1) // 2)

        # 3) Patch Merging(또는 Identity) 직후 SB 연속 적용
        for sb in self.pre_sbs:
            hs = sb(hs, new_hw, None, None, False)[0]

        # 4) 원래 downsample의 반환 '형식'을 그대로 보존해서 돌려줌
        if isinstance(ds_out, tuple):
            out_list = list(ds_out)
            out_list[0] = hs
            # 해상도를 튜플로 돌려주는 버전이면 갱신된 new_hw를 넣어줌
            if len(out_list) >= 2 and isinstance(out_list[1], tuple):
                out_list[1] = new_hw
            return tuple(out_list)
        else:
            return hs


def insert_spectral_before_stage_blocks(stages, *, config, sb_insert_spec: str):
    """
    Stage별 블록 '시작 전에' SB를 k개 삽입:
      S1: Linear Embedding → SB×k → SwinBlock×2
      S2: Patch Merging → SB×k → SwinBlock×2
      S3: Patch Merging → SB×k → SwinBlock×6
      S4: Patch Merging → SB×k → SwinBlock×2
    """
    depths = list(getattr(config, "depths"))
    plan = parse_sb_insert_spec(sb_insert_spec, len(depths))
    if not plan:
        print("[SB-INSERT] 지정 없음 (원본 Swin 유지).")
        return

    for s0, count in sorted(plan.items()):
        stage = stages[s0]
        # 각 stage의 입력 채널 (= 블록들이 받는 dim)
        dim = config.embed_dim * (2 ** s0)
        heads = config.num_heads[s0] if isinstance(config.num_heads, (list, tuple)) else config.num_heads
        mlp_r = getattr(config, "mlp_ratio", 4.0)

        # stage 시작부에서 먼저 통과시킬 SB 리스트
        pre_sbs = nn.ModuleList([
            HFSwinBlockAdapter(
                dim=dim, num_heads=heads, mlp_ratio=mlp_r, sb_id=f"S{s0+1}I{k+1}"
            ) for k in range(count)
        ])
        # stage 객체에 보관
        stage._pre_sbs = pre_sbs

        # 원래 forward 백업(한 번만)
        if not hasattr(stage, "_orig_forward"):
            stage._orig_forward = stage.forward

        # 새 forward: SB들 → 원래 stage.forward(블록들+stage 내부 로직)
        def new_forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False):
            # SB들은 head_mask 없이 선적용
            for sb in self._pre_sbs:
                hidden_states = sb(hidden_states, input_dimensions, None, None, False)[0]
            # 이후는 원래 stage 로직 (블록들 루프, 필요시 stage 내부 downsample 등)
            return self._orig_forward(hidden_states, input_dimensions, head_mask, output_attentions)

        # 메서드 바인딩
        stage.forward = types.MethodType(new_forward, stage)
        print(f"[SB-INSERT] Stage {s0+1}: 블록 시작 전에 SB ×{count} 삽입 (dim={dim})")


class MedCLIPVisionModelViT_SGN_Inserted(MedCLIPVisionModelViT):
    """Swin 시작부 삽입 모드"""
    def __init__(
        self, checkpoint=None, medclip_checkpoint=None,
        # 둘 다 받되, sb_insert_spec 우선 사용
        sb_insert_spec: str = "",
        sb_spec: str = ""
    ):
        super().__init__(checkpoint=checkpoint, medclip_checkpoint=medclip_checkpoint)
        hf_model = self.model
        if not (hasattr(hf_model, "encoder") and hasattr(hf_model.encoder, "layers")):
            raise NotImplementedError("Swin 백본만 지원합니다.")
        cfg, stages = hf_model.config, hf_model.encoder.layers
        plan_spec = sb_insert_spec if sb_insert_spec else sb_spec  # 호환
        insert_spectral_before_stage_blocks(stages, config=cfg, sb_insert_spec=plan_spec)
        print(f"[SGN-Swin INSERT] '{plan_spec}' 적용 (embed_dim={cfg.embed_dim}, depths={cfg.depths})")


class MedCLIPBinaryClassifier_SGN_Inserted(nn.Module):
    def __init__(
        self, num_classes: int = 2, dropout: float = 0.0, freeze_backbone="OFF",
        # 둘 다 받되, sb_insert_spec 우선
        sb_insert_spec: str = "",
        sb_spec: str = ""
    ):
        super().__init__()
        plan_spec = sb_insert_spec if sb_insert_spec else sb_spec  # 호환
        self.backbone = MedCLIPVisionModelViT_SGN_Inserted(sb_insert_spec=plan_spec)
        cfg = self.backbone.model.config
        embed_dim = getattr(cfg, "hidden_size", cfg.embed_dim * (2 ** (len(cfg.depths) - 1)))
        _apply_freeze_policy(self.backbone, freeze_backbone)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values=x, project=False)
        return self.head(self.dropout(feats))
