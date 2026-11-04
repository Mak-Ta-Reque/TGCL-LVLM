"""How a segment is shown to the VLM (steps 3 and 4).

A segment of crops.json (a mask and/or box on an image) becomes the image the
model answers: cut with context, the outside of the mask inpainted, and fitted
into the patch size. Shared by feature saving (tgcl.save_features), layer
selection and the segment probe used during development.
"""

import functools
import os
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image
from PIL import ImageFilter
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion

from tgcl.imputers import noisy_linear_inpaint

_LANCZOS = Image.Resampling.LANCZOS


def resize_to_patch(image: Image.Image, patch: Optional[int]) -> Image.Image:
    """A crop as the VLM sees it (steps 3 and 4).

    Default (CROP_RESIZE=pad): scaled, aspect kept, to fit patch x patch and
    centred on a white patch x patch square. CROP_RESIZE=none: the crop as cut,
    unchanged; the model's own processor sizes it (Gemma-3n: 768 x 768;
    Qwen2.5-VL: native size within QWEN_MIN_PIXELS..QWEN_MAX_PIXELS), so the
    object is not shrunk into a white frame.
    """
    if (
        os.environ.get("CROP_RESIZE", "pad") == "none"
        or not patch
        or patch <= 0
    ):
        return image
    resample = _LANCZOS
    width, height = image.size
    scale = min(patch / width, patch / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    image = image.resize((new_w, new_h), resample=resample)
    if new_w != patch or new_h != patch:
        padded = Image.new("RGB", (patch, patch), (255, 255, 255))
        padded.paste(image, ((patch - new_w) // 2, (patch - new_h) // 2))
        image = padded
    return image


# ---------------------------------------------------------------------------
# Background masking helper — replaces pixels outside the segment mask
# ---------------------------------------------------------------------------


def apply_background_masking(
    crop_img: Image.Image,
    crop_mask_np,  # bool ndarray (H, W)
    method: str = "gaussian_blur",
    blur_radius: int = 15,
) -> Image.Image:
    """Apply background masking to a cropped image.

    Args:
        crop_img:      PIL RGB image (the tight-bbox crop).
        crop_mask_np:  Boolean numpy array, True = foreground (keep sharp).
        method:        One of ``'gaussian_blur'``, ``'ns'`` (Navier-Stokes
                       inpainting), ``'telea'`` (Telea inpainting),
                       ``'noisy_linear'`` (sparse linear interpolation),
                       or ``'blackout'`` (zero-fill).
        blur_radius:   Radius for Gaussian blur *or* inpainting radius.

    Returns:
        PIL RGB image with background masked according to *method*.
    """
    crop_mask_pil = Image.fromarray(
        crop_mask_np.astype("uint8") * 255, mode="L"
    )

    if method == "gaussian_blur":
        blurred = crop_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.composite(crop_img, blurred, crop_mask_pil)

    elif method in ("ns", "telea"):
        # OpenCV Navier-Stokes or Telea inpainting
        import cv2

        img_array = np.array(crop_img)  # RGB uint8 (H, W, 3)
        # Inpainting mask: 255 where we want to inpaint (i.e. outside fg)
        inpaint_mask = (~crop_mask_np).astype("uint8") * 255
        flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
        # cv2.inpaint expects BGR input
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(
            img_bgr,
            inpaint_mask,
            inpaintRadius=max(3, blur_radius // 5),
            flags=flag,
        )
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        # Composite: keep original foreground pixels, use inpainted for bg
        inpainted_pil = Image.fromarray(inpainted_rgb)
        return Image.composite(crop_img, inpainted_pil, crop_mask_pil)

    elif method == "noisy_linear":
        # Noisy linear imputation (Rong et al. / ROAD framework)
        inpainted_pil = noisy_linear_inpaint(crop_img, crop_mask_np, noise=0.01)
        # Composite: keep original fg sharp, use imputed values for bg
        return Image.composite(crop_img, inpainted_pil, crop_mask_pil)

    elif method == "blackout":
        black = Image.new("RGB", crop_img.size, (0, 0, 0))
        return Image.composite(crop_img, black, crop_mask_pil)

    else:
        # Unknown method — fall back to Gaussian blur
        blurred = crop_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.composite(crop_img, blurred, crop_mask_pil)


def apply_boundary_smoothing(
    image: Image.Image,
    mask_np,  # bool ndarray (H, W)
    boundary_pixels: int,
) -> Image.Image:
    """Smooth the mask-edge band to reduce silhouette leakage."""
    if boundary_pixels <= 0:
        return image

    struct = np.ones(
        (2 * boundary_pixels + 1, 2 * boundary_pixels + 1), dtype=bool
    )
    outer = binary_dilation(mask_np, structure=struct, iterations=1)
    inner = binary_erosion(mask_np, structure=struct, iterations=1)
    boundary_band = np.logical_and(outer, np.logical_not(inner))
    if not boundary_band.any():
        return image

    src = np.array(image, dtype=np.float32)
    blurred = np.array(
        image.filter(
            ImageFilter.GaussianBlur(radius=max(1, boundary_pixels // 2))
        ),
        dtype=np.float32,
    )

    alpha = np.zeros(mask_np.shape, dtype=np.float32)
    alpha[boundary_band] = 1.0
    alpha = np.expand_dims(alpha, axis=-1)

    out = src * (1.0 - alpha) + blurred * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def run_inpainting_step(
    crop_img: Image.Image,
    crop_mask_np,  # bool ndarray (H, W)
    method: str,
    blur_radius: int,
    boundary_pixels: int,
) -> Image.Image:
    """Single inpainting entry point used by save_features mask pipeline."""
    if blur_radius > 0 or method != "gaussian_blur":
        vlm_img = apply_background_masking(
            crop_img,
            crop_mask_np,
            method=method,
            blur_radius=blur_radius,
        )
    else:
        vlm_img = crop_img

    return apply_boundary_smoothing(vlm_img, crop_mask_np, boundary_pixels)


def masked_region_view(
    img: Image.Image,
    mask_np,  # bool ndarray (H, W), same size as img
    *,
    is_concept: bool,
    method: str,
    blur_radius: int,
    context_pixels: int,
    boundary_pixels: int,
    patch_size: Optional[int],
    full_image: bool = False,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """VLM input for one segmentation mask; returns (image, region box
    x1,y1,x2,y2).

    Default: crop to the mask's box (+ context pixels), fill outside the mask
    with `method`, resize to patch_size
    (resize_to_patch:
    white padding by default, CROP_RESIZE=none keeps the crop as cut).

    full_image=True (MASK_FULL_IMAGE=1): keep the whole image, fill outside the
    mask with `method` (e.g. "blackout"), and skip the patch resize so the
    input has the same geometry as whole-image (CROP_MODE=none) inputs.
    """
    H, W = mask_np.shape
    ys, xs = np.where(mask_np)
    if len(ys) == 0:
        raise ValueError("Mask is empty (0 foreground pixels)")

    if full_image:
        x1, y1, x2, y2 = 0, 0, W, H
    else:
        # Expand the mask box by context + boundary pixels so that a dilated
        # FG mask still fits inside the crop.
        pad = max(context_pixels, 0, boundary_pixels)
        x1 = max(0, min(W, int(xs.min()) - pad))
        y1 = max(0, min(H, int(ys.min()) - pad))
        x2 = max(0, min(W, int(xs.max()) + 1 + pad))
        y2 = max(0, min(H, int(ys.max()) + 1 + pad))
    region_img = img.crop((x1, y1, x2, y2))
    region_mask = mask_np[y1:y2, x1:x2]

    # Shared boundary-width adjustment. FG (is_concept=True) -> dilate: keep a
    # ring of real surrounding pixels as spatial context. BG -> erode: remove
    # pixels near the object edge so the silhouette does not leak into the
    # background view. The same width randomizes pixels at the mask edge.
    if boundary_pixels > 0:
        struct = np.ones(
            (2 * boundary_pixels + 1, 2 * boundary_pixels + 1), dtype=bool
        )
        if is_concept:
            region_mask = binary_dilation(
                region_mask, structure=struct, iterations=1
            ).astype(bool)
        else:
            eroded = binary_erosion(region_mask, structure=struct, iterations=1)
            # Keep the original mask if erosion empties it
            if eroded is not None and eroded.any():
                region_mask = eroded.astype(bool)

    vlm_img = run_inpainting_step(
        region_img,
        region_mask,
        method=method,
        blur_radius=blur_radius,
        boundary_pixels=boundary_pixels,
    )

    if not full_image:
        vlm_img = resize_to_patch(vlm_img, patch_size)
    return vlm_img, (x1, y1, x2, y2)


def _target_size(size: Any) -> Optional[Tuple[int, int]]:
    """crops.json image_size [w, h] as a tuple, or None (keep the image's own
    size)."""
    try:
        w, h = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (w, h) if w > 0 and h > 0 else None


@functools.lru_cache(maxsize=512)
def _load_resized(path: str, size: Optional[Tuple[int, int]]) -> Image.Image:
    """Decoded RGB image at the crops.json image size. Cached (callers only crop
    it): a class has ~900 segments from ~300 images, and decoding plus
    LANCZOS-resizing the full image for every segment was the slowest CPU part
    of step 4 (8.7 ms vs 0.15 ms per crop).
    """
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(
            size,
            resample=_LANCZOS,
        )
    return img


def segment_image(img_ref: Any, size: Any) -> Image.Image:
    """The image a segment is cut from, at its crops.json image size."""
    size = _target_size(size) if isinstance(size, (list, tuple)) else None
    if isinstance(img_ref, Image.Image):
        if size is not None and img_ref.size != size:
            return img_ref.resize(
                size,
                resample=_LANCZOS,
            )
        return img_ref
    return _load_resized(str(img_ref), size)


MAX_SEGMENT_SOURCE_SIDE = 1024


def segment_source(img_ref: Any, size: Any) -> Tuple[Image.Image, float]:
    """The image a segment is cut from, and its scale relative to the crops.json
    frame (masks, boxes and context pixels are given in that frame).

    SEGMENT_RESOLUTION=frame (default): the image resized to the frame (step
    2's working size), scale 1.
    SEGMENT_RESOLUTION=original: the image at its own resolution (long side at
    most MAX_SEGMENT_SOURCE_SIDE), so a small object keeps its pixels;
    a development probe measured +14 points recall for Gemma-3n on COCO
    objects (+21 on objects under 40 px in the frame).
    """
    frame = _target_size(size) if isinstance(size, (list, tuple)) else None
    if (
        os.environ.get("SEGMENT_RESOLUTION", "frame") != "original"
        or frame is None
    ):
        return segment_image(img_ref, size), 1.0
    if isinstance(img_ref, Image.Image):
        w, h = img_ref.size
    else:
        with Image.open(str(img_ref)) as im:
            w, h = im.size
    shrink = min(1.0, MAX_SEGMENT_SOURCE_SIDE / max(w, h))
    target = (max(1, round(w * shrink)), max(1, round(h * shrink)))
    img = segment_image(img_ref, list(target))
    return img, img.width / frame[0]
