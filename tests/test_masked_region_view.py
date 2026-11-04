"""Segment input construction: helpers.segment_view.masked_region_view."""

import numpy as np
from PIL import Image
import pytest

from tgcl.helpers.segment_view import masked_region_view
from tgcl.helpers.segment_view import run_inpainting_step

W, H = 80, 60


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return Image.fromarray(
        rng.integers(30, 255, size=(H, W, 3), dtype=np.uint8)
    )


@pytest.fixture
def mask():
    m = np.zeros((H, W), dtype=bool)
    m[20:35, 30:55] = True
    return m


def reference_crop_view(
    img, mask_np, is_concept, method, blur, context, boundary, patch
):
    """The crop path as it was inline in save_features.py before refactoring."""
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import binary_erosion

    ys, xs = np.where(mask_np)
    mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

    def clip(v, lo, hi):
        return max(lo, min(hi, v))

    pad = max(context, 0, boundary)
    cx1, cy1 = clip(mx1 - pad, 0, W), clip(my1 - pad, 0, H)
    cx2, cy2 = clip(mx2 + pad, 0, W), clip(my2 + pad, 0, H)
    crop_img = img.crop((cx1, cy1, cx2, cy2))
    crop_mask = mask_np[cy1:cy2, cx1:cx2]
    if boundary > 0:
        struct = np.ones((2 * boundary + 1, 2 * boundary + 1), dtype=bool)
        if is_concept:
            crop_mask = binary_dilation(
                crop_mask, structure=struct, iterations=1
            ).astype(bool)
        else:
            eroded = binary_erosion(crop_mask, structure=struct, iterations=1)
            if eroded.any():
                crop_mask = eroded.astype(bool)
    out = run_inpainting_step(
        crop_img,
        crop_mask,
        method=method,
        blur_radius=blur,
        boundary_pixels=boundary,
    )
    cw, ch = out.size
    scale = min(patch / cw, patch / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    out = out.resize((nw, nh), resample=Image.Resampling.LANCZOS)
    if (nw, nh) != (patch, patch):
        padded = Image.new("RGB", (patch, patch), (255, 255, 255))
        padded.paste(out, ((patch - nw) // 2, (patch - nh) // 2))
        out = padded
    return out, (int(cx1), int(cy1), int(cx2), int(cy2))


@pytest.mark.parametrize("is_concept", [True, False])
@pytest.mark.parametrize(
    "method, context", [("blackout", 0), ("gaussian_blur", 0), ("blackout", 3)]
)
def test_crop_path_unchanged_by_refactor(
    image, mask, is_concept, method, context
):
    np.random.seed(0)  # boundary smoothing randomizes edge pixels
    expected, expected_box = reference_crop_view(
        image, mask, is_concept, method, 10, context, context, 40
    )
    np.random.seed(0)
    actual, box = masked_region_view(
        image,
        mask,
        is_concept=is_concept,
        method=method,
        blur_radius=10,
        context_pixels=context,
        boundary_pixels=context,
        patch_size=40,
    )
    assert box == expected_box
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_full_image_blackout_keeps_geometry_and_object(image, mask):
    out, box = masked_region_view(
        image,
        mask,
        is_concept=False,
        method="blackout",
        blur_radius=10,
        context_pixels=0,
        boundary_pixels=0,
        patch_size=40,
        full_image=True,
    )
    pixels, original = np.asarray(out), np.asarray(image)
    assert out.size == (W, H)  # no crop, no patch resize
    assert box == (0, 0, W, H)
    assert np.array_equal(pixels[mask], original[mask])
    assert not pixels[~mask].any()  # everything outside the mask is black


def test_empty_mask_is_rejected(image):
    with pytest.raises(ValueError, match="empty"):
        masked_region_view(
            image,
            np.zeros((H, W), dtype=bool),
            is_concept=True,
            method="blackout",
            blur_radius=0,
            context_pixels=0,
            boundary_pixels=0,
            patch_size=40,
        )
