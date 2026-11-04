"""datasets.image_text_dataset.select_segments: which BAG_SIZE segments of a
class step 4 asks about."""

import numpy as np

from tgcl.data.image_text_dataset import select_segments

ITEMS = [
    {"id": i, "bbox": [0, 0, side, side]}
    for i, side in enumerate([10, 300, 40, 200, 5, 120])
]


def test_largest_keeps_the_biggest_boxes(monkeypatch):
    monkeypatch.setenv("SEGMENT_SELECTION", "largest")
    chosen = select_segments(ITEMS, 3, np.random.default_rng(0))
    assert [c["id"] for c in chosen] == [1, 3, 5]


def test_random_is_the_default_and_seeded(monkeypatch):
    monkeypatch.delenv("SEGMENT_SELECTION", raising=False)
    import random

    random.seed(0)
    a = [c["id"] for c in select_segments(ITEMS, 3, np.random.default_rng(1))]
    random.seed(0)
    b = [c["id"] for c in select_segments(ITEMS, 3, np.random.default_rng(1))]
    assert a == b and len(set(a)) == 3


def test_fewer_items_than_the_bag_are_all_kept(monkeypatch):
    monkeypatch.setenv("SEGMENT_SELECTION", "largest")
    assert len(select_segments(ITEMS, 10, np.random.default_rng(0))) == len(
        ITEMS
    )
