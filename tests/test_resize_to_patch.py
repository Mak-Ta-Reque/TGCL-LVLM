"""helpers.segment_view.resize_to_patch: the crop the VLM sees in steps 3 and
4."""

import numpy as np
from PIL import Image

from tgcl.helpers.segment_view import resize_to_patch

CROP = Image.new("RGB", (40, 20), (10, 20, 30))  # a small, wide segment


def test_default_pads_to_a_white_square(monkeypatch):
    monkeypatch.delenv("CROP_RESIZE", raising=False)
    out = resize_to_patch(CROP, 160)
    assert out.size == (160, 160)
    arr = np.asarray(out)
    assert (arr[0, 0] == 255).all() and tuple(arr[80, 80]) == (10, 20, 30)


def test_none_passes_the_crop_unchanged(monkeypatch):
    monkeypatch.setenv("CROP_RESIZE", "none")
    assert resize_to_patch(CROP, 160) is CROP
