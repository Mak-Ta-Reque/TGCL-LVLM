"""LangSAM batching: an out-of-memory batch falls back to single images; an
image that still runs out of memory is skipped; other errors are raised."""

import numpy as np
from PIL import Image
import pytest
import torch

# Skipped where lang_sam (which the module imports) is not installed.
predict_bboxes_and_masks_for_tag_batched = pytest.importorskip(
    "tgcl.langsam_utils"
).predict_bboxes_and_masks_for_tag_batched


def _result(width, height):
    mask = np.zeros((1, height, width), dtype=bool)
    mask[0, 2:8, 3:9] = True
    return {"boxes": np.array([[3.0, 2.0, 9.0, 8.0]]), "masks": mask}


class _FakeLangSAM:
    """Answers one box per image; runs out of memory on batches and on the
    images whose width is in oom_widths."""

    def __init__(self, oom_widths=(), error=None):
        self.oom_widths = set(oom_widths)
        self.error = error
        self.calls = []

    def predict(self, images, tags):
        self.calls.append(len(images))
        if self.error is not None:
            raise self.error
        if len(images) > 1 or images[0].size[0] in self.oom_widths:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return [_result(*im.size) for im in images]


def _images():
    return [Image.new("RGB", (20 + i, 16)) for i in range(5)]


def test_oom_batch_falls_back_to_single_images():
    model = _FakeLangSAM()
    pairs = predict_bboxes_and_masks_for_tag_batched(
        model, _images(), "cup", batch_size=4
    )
    assert model.calls == [4, 1, 1, 1, 1, 1]
    assert [len(p) for p in pairs] == [1] * 5
    assert pairs[0][0][0] == [3, 2, 6, 6]


def test_image_that_still_runs_out_of_memory_is_skipped():
    model = _FakeLangSAM(oom_widths={21})
    pairs = predict_bboxes_and_masks_for_tag_batched(
        model, _images(), "cup", batch_size=4
    )
    assert [len(p) for p in pairs] == [1, 0, 1, 1, 1]


def test_other_errors_are_raised():
    model = _FakeLangSAM(error=ValueError("bad input"))
    with pytest.raises(ValueError):
        predict_bboxes_and_masks_for_tag_batched(model, _images(), "cup")
