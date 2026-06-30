# sci/data_preprocessing.py
from typing import Optional, Union, List, Tuple
import numpy as np

# Use OpenCV when available; otherwise fall back to scipy.
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    # scipy fallback operations: Gaussian filtering, morphology, and labeling.
    from scipy.ndimage import (
        gaussian_filter, binary_opening, binary_closing, binary_fill_holes,
        label as ndi_label, generate_binary_structure
    )  # type: ignore

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821
Polygon = List[Tuple[float, float]]            # [(x,y), ...]


# ------------------------------- conversion utilities -------------------------------

def _to_numpy_uint8(img: Union[ArrayLike, "Image.Image"]) -> np.ndarray:
    """Convert img to np.uint8 with shape (H,W,C) or (H,W)."""
    if _HAS_TORCH and isinstance(img, torch.Tensor):
        x = img.detach().cpu().numpy()
        # (C,H,W) or (H,W).
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        # Infer value scale.
        if x.dtype.kind == 'f' and x.max() <= 1.0:
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        elif x.dtype != np.uint8:
            x = x.clip(0, 255).astype(np.uint8)
        return x
    if _HAS_PIL and isinstance(img, Image.Image):
        return np.array(img.convert("L" if img.mode == "L" else "RGB"), dtype=np.uint8)
    # numpy
    x = np.asarray(img)
    if x.dtype != np.uint8:
        if x.dtype.kind == 'f' and x.max() <= 1.0:
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x = x.clip(0, 255).astype(np.uint8)
    # (C,H,W) -> (H,W,C)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[0] < x.shape[2]:
        x = np.transpose(x, (1, 2, 0))
    return x


def _from_numpy(img_np: np.ndarray, like: Union[ArrayLike, "Image.Image"]) -> Union[ArrayLike, "Image.Image"]:
    """Convert back to the original type."""
    if _HAS_TORCH and isinstance(like, torch.Tensor):
        x = img_np.astype(np.float32) / 255.0
        if x.ndim == 2:  # (H,W) -> (1,H,W)
            x = x[None, ...]
        else:            # (H,W,C) -> (C,H,W)
            x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x).type_as(like)
    if _HAS_PIL and isinstance(like, Image.Image):
        mode = "L" if img_np.ndim == 2 else "RGB"
        return Image.fromarray(img_np, mode=mode)
    return img_np


def _ensure_odd(v: int) -> int:
    return v if (v % 2 == 1) else (v + 1)


def _tensor_to_u8_gray(x: Union[ArrayLike, "Image.Image"]) -> np.ndarray:
    """
    Convert (1,H,W) torch float[0,1] or (H,W)/(H,W,C) uint8/float input to grayscale uint8(H,W).
    """
    arr = _to_numpy_uint8(x)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 1:
        return arr[..., 0]
    if _HAS_CV2:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Without cv2, use a simple channel mean.
    return arr.mean(axis=2).astype(np.uint8)


# ------------------------------- mask utilities -------------------------------

def _rasterize_annotation(annotation, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert annotation to an (H,W) binary mask.
    - If annotation is already an (H,W) ndarray, use it directly; 0/1 and 0/255 are accepted.
    - Convert torch.Tensor annotations to numpy.
    - Rasterize polygon annotations, either list[(x,y)] or list[list[(x,y)]].
    """
    H, W = shape

    if annotation is None:
        return None

    # ndarray / tensor
    if _HAS_TORCH and isinstance(annotation, torch.Tensor):
        annotation = annotation.detach().cpu().numpy()
    if isinstance(annotation, np.ndarray):
        m = annotation
        if m.ndim == 3 and m.shape[0] in (1,):   # (1,H,W) -> (H,W)
            m = m[0]
        if m.ndim == 3 and m.shape[-1] == 1:     # (H,W,1) -> (H,W)
            m = m[..., 0]
        if m.shape != (H, W):
            raise ValueError(f"annotation mask shape {m.shape} != image shape {(H, W)}")
        m = (m > 0).astype(np.uint8)
        return m

    # polygon(s)
    polys: List[np.ndarray] = []
    if isinstance(annotation, (list, tuple)) and len(annotation) > 0:
        # list of points or list of list-of-points
        if isinstance(annotation[0], (list, tuple)) and len(annotation[0]) > 0 and isinstance(annotation[0][0], (int, float)):
            # [(x,y), ...]
            polys = [np.array(annotation, dtype=np.int32)]
        else:
            # [ [(x,y),...], [(x,y),...] ]
            for poly in annotation:
                polys.append(np.array(poly, dtype=np.int32))
    if polys:
        if not _HAS_CV2:
            raise RuntimeError("cv2 is unavailable, so polygon rasterization cannot be performed. Pass annotation as a binary mask.")
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, polys, 1)
        return mask

    raise ValueError("Unrecognized annotation format. Pass a binary mask or polygon.")


def _otsu_threshold(gray_u8: np.ndarray) -> int:
    """Simple approximate Otsu threshold for environments without cv2."""
    hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256))
    prob = hist.astype(np.float64) / gray_u8.size
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    sigma_b2[~np.isfinite(sigma_b2)] = -1
    return int(np.argmax(sigma_b2))


def build_auto_mask_from_image_u8(
    gray_u8: np.ndarray,
    method: str = "otsu",
    percentile: float = 70.0,
    keep_components: int = 2,
    min_area_ratio: float = 1e-3,
) -> np.ndarray:
    """
    gray_u8: (H,W) uint8
    return: (H,W) uint8 binary mask, 1=lung region, usually darker, 0=background
    """
    H, W = gray_u8.shape

    # 1) Initial thresholding, assuming lung tissue is darker.
    if method == "otsu":
        if _HAS_CV2:
            _, bin_img = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            mask = (bin_img > 0).astype(np.uint8)
        else:
            thr = _otsu_threshold(gray_u8)
            mask = (gray_u8 <= thr).astype(np.uint8)
    else:  # percentile
        thr = float(np.percentile(gray_u8, float(percentile)))
        mask = (gray_u8 <= thr).astype(np.uint8)

    # 2) Morphological refinement.
    if _HAS_CV2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    else:
        st = generate_binary_structure(2, 1)
        mask = binary_closing(mask.astype(bool), structure=st, iterations=2)
        mask = binary_opening(mask, structure=st, iterations=1)
        mask = binary_fill_holes(mask)
        mask = mask.astype(np.uint8)

    # 3) Keep the largest connected components.
    min_area = max(1, int(min_area_ratio * (H * W)))
    if _HAS_CV2:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            items = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
            items.sort(key=lambda t: t[1], reverse=True)
            keep_ids = {i for i, _ in items[:max(1, int(keep_components))]}
            out = np.zeros_like(mask, dtype=np.uint8)
            for i in keep_ids:
                out[labels == i] = 1
            mask = out
    else:
        st = generate_binary_structure(2, 1)
        labels, num = ndi_label(mask.astype(bool), structure=st)
        if num > 0:
            areas = [(i, int((labels == i).sum())) for i in range(1, num + 1)]
            areas = [(i, a) if a >= min_area else (i, 0) for i, a in areas]
            areas.sort(key=lambda t: t[1], reverse=True)
            keep_ids = {i for i, a in areas[:max(1, int(keep_components))] if a > 0}
            out = np.zeros_like(mask, dtype=np.uint8)
            for i in keep_ids:
                out[labels == i] = 1
            mask = out

    return mask.astype(np.uint8)


# ------------------------------- blur core -------------------------------

def apply_blur_with_annotation(
    img: Union[ArrayLike, "Image.Image"],
    annotation: Optional[Union[np.ndarray, "torch.Tensor", Polygon, List[Polygon]]] = None,
    ksize: int = 31,
    sigma: float = 0.0,
    feather_px: int = 8,
    bright_only: bool = True,
    bright_percentile: float = 98.0,
    # === automatic mask options, used when annotation is None ===
    auto_when_none: bool = True,
    auto_method: str = "otsu",               # ["otsu", "percentile"]
    auto_percentile: float = 70.0,           # Threshold used when auto_method="percentile".
    auto_keep_components: int = 2,
    auto_min_area_ratio: float = 1e-3,
):
    """
    Softly blur bright regions outside the lung mask annotation.
    - img: PIL, numpy, and torch tensor inputs are supported.
    - annotation: (H,W) binary mask with 1=lung region, or polygon(s).
    - ksize: Gaussian kernel size; odd values are recommended.
    - sigma: if 0, OpenCV computes it from ksize.
    - feather_px: boundary feathering width in pixels.
    - bright_only: if True, blur only bright pixels outside the mask to preserve background.
    - bright_percentile: bright-pixel threshold as the p-th percentile outside the mask.
    - auto_when_none: build an automatic mask from the image when annotation is unavailable.
    - auto_method/percentile/keep_components/min_area_ratio: automatic-mask parameters.

    Return the same type as the input.
    """
    # 1) Convert to numpy and prepare grayscale data.
    like = img
    x = _to_numpy_uint8(img)  # (H,W) or (H,W,C).
    if x.ndim == 2:
        gray = x
    elif x.ndim == 3 and x.shape[2] == 1:
        gray = x[..., 0]  # Already single-channel.
    elif _HAS_CV2:
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if x.shape[2] == 3 else cv2.cvtColor(x, cv2.COLOR_RGBA2GRAY)
    else:
        gray = x[..., 0]

    # 2) annotation -> mask, where 1=lung region.
    mask_in: Optional[np.ndarray] = None
    if annotation is None:
        if auto_when_none:
            # Build an automatic mask from the image itself.
            mask_in = build_auto_mask_from_image_u8(
                gray,
                method=auto_method,
                percentile=auto_percentile,
                keep_components=auto_keep_components,
                min_area_ratio=auto_min_area_ratio,
            )
        else:
            # If no mask is available, return the image unchanged.
            return img
    else:
        mask_in = _rasterize_annotation(annotation, (H, W)).astype(np.uint8)  # 1=lung, 0=non-lung

    # mask_out: region where blur is applied.
    mask_out = (1 - mask_in).astype(np.uint8)

    # 3) Create the blurred image.
    ksize = _ensure_odd(int(ksize))
    if _HAS_CV2:
        blurred = cv2.GaussianBlur(x, (ksize, ksize), sigma)
    else:
        # Channel-wise Gaussian filtering.
        if x.ndim == 2:
            blurred = gaussian_filter(x, sigma=max(1.0, ksize / 6.0))
        else:
            blurred = np.stack(
                [gaussian_filter(x[..., c], sigma=max(1.0, ksize / 6.0)) for c in range(x.shape[2])],
                axis=-1
            )
        blurred = blurred.astype(np.uint8)

    # 4) Optionally select only bright regions outside the mask.
    if bright_only:
        outside_vals = gray[mask_out.astype(bool)]
        if outside_vals.size > 0:
            thr = np.percentile(outside_vals, float(bright_percentile))
        else:
            thr = 255
        bright_mask = (gray >= thr).astype(np.uint8)
        target = (mask_out & bright_mask).astype(np.uint8)
    else:
        target = mask_out

    # 5) Boundary feathering with soft alpha.
    if feather_px > 0:
        if _HAS_CV2:
            soft = cv2.GaussianBlur((target * 255).astype(np.uint8), (0, 0), float(feather_px)).astype(np.float32) / 255.0
        else:
            soft = gaussian_filter(target.astype(np.float32), sigma=float(feather_px))
            mmin, mmax = float(soft.min()), float(soft.max())
            soft = (soft - mmin) / (mmax - mmin + 1e-6)
    else:
        soft = target.astype(np.float32)

    # 6) Composite x*(1-a) + blurred*a.
    if x.ndim == 2:
        a = soft
        out = (x.astype(np.float32) * (1.0 - a) + blurred.astype(np.float32) * a).round().clip(0, 255).astype(np.uint8)
    else:
        a = soft[..., None]  # (H,W,1)
        out = (x.astype(np.float32) * (1.0 - a) + blurred.astype(np.float32) * a).round().clip(0, 255).astype(np.uint8)

    # 7) Restore the original type.
    return _from_numpy(out, like)
