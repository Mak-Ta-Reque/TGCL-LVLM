"""Crop boxes of crops.json and the grounding segments of a concept bank, as
step 4 saw them.

preprocessing/crops_to_json.py resizes every image to IMAGE_SIZE_WIDTH (aspect
ratio kept, small images upscaled) and stores each segment's box as [x, y, w, h]
in that resized image. The concept bank keeps the original image path and that
box, so a grounding crop is: open, resize to the same width, crop the box.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from PIL import Image

DEFAULT_IMAGE_WIDTH = 512  # IMAGE_SIZE_WIDTH of the pipeline


def bbox_xywh_to_xyxy(
    bbox: Sequence[float], width: int, height: int
) -> Tuple[int, int, int, int]:
    """Crop box (x1, y1, x2, y2) of a crops-JSON "bbox" [x, y, w, h], clipped to
    the image, at least 1 px.

    Reading the box as [x1, y1, x2, y2] (as step 4 did before 2026-10-21) turns
    a sliding window [192, 0, 160, 160] into a 1-pixel strip.
    """
    x, y, w, h = (float(v) for v in bbox)
    x1 = int(min(max(x, 0), width - 1))
    y1 = int(min(max(y, 0), height - 1))
    x2 = int(min(max(x + w, x1 + 1), width))
    y2 = int(min(max(y + h, y1 + 1), height))
    return x1, y1, x2, y2


def resize_to_width(img: Image.Image, width: Optional[int]) -> Image.Image:
    """The image as crops_to_json resized it (_compute_virtual_resize): `width`
    wide, aspect kept."""
    w, h = img.size
    if not width or width <= 0 or w <= 0 or (w == width):
        return img
    new_h = max(1, int(round(h * width / float(w))))
    resample = getattr(
        getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC
    )
    return img.resize((int(width), new_h), resample=resample)


def grounding_crop(
    path: Union[str, Path],
    bbox: Optional[Sequence[float]],
    image_width: Optional[int] = DEFAULT_IMAGE_WIDTH,
) -> Image.Image:
    """Grounding segment of a concept: the image at `image_width`, cropped to
    bbox [x, y, w, h].

    Without a box (whole-image grounding) the resized image is returned.
    """
    img = resize_to_width(Image.open(path).convert("RGB"), image_width)
    if bbox is None or len(bbox) != 4:
        return img
    return img.crop(bbox_xywh_to_xyxy(bbox, *img.size))
