#!/usr/bin/env python3
"""
Unit tests for crops_to_json.py geometry helpers (CPU only, no detector).

Run with:
    python -m pytest tests/test_crops_to_json_unit.py
"""

import unittest

import numpy as np

from tgcl.preprocessing.crops_to_json import _compute_virtual_resize
from tgcl.preprocessing.crops_to_json import (
    _greedy_filter_candidates_by_clip_similarity,
)
from tgcl.preprocessing.crops_to_json import _sliding_window_bboxes_for_image
from tgcl.preprocessing.crops_to_json import _subtract_mask


class TestComputeVirtualResize(unittest.TestCase):
    """Test virtual resize computation."""

    def test_no_resize_when_none(self):
        w, h, scale = _compute_virtual_resize(800, 600, None)
        self.assertEqual((w, h), (800, 600))
        self.assertAlmostEqual(scale, 1.0)

    def test_downscale(self):
        w, h, scale = _compute_virtual_resize(1000, 500, 500)
        self.assertEqual((w, h), (500, 250))
        self.assertAlmostEqual(scale, 0.5)

    def test_upscale(self):
        w, h, scale = _compute_virtual_resize(200, 100, 400)
        self.assertEqual((w, h), (400, 200))
        self.assertAlmostEqual(scale, 2.0)

    def test_preserves_aspect_ratio(self):
        new_w, new_h, _ = _compute_virtual_resize(1920, 1080, 640)
        self.assertAlmostEqual(1920 / 1080, new_w / new_h, places=2)

    def test_zero_width_returns_original(self):
        w, h, _ = _compute_virtual_resize(800, 600, 0)
        self.assertEqual((w, h), (800, 600))


class TestClipRandomCropFilter(unittest.TestCase):
    """CLIP-based greedy near-duplicate filtering."""

    def test_filters_similar_embeddings(self):
        # Candidates 0 and 1 are near-duplicates (cosine 0.96); 2 is distinct.
        # The candidate order is shuffled, so either duplicate may survive.
        candidate_pairs = [
            ([0, 0, 10, 10], None),
            ([20, 20, 10, 10], None),
            ([40, 40, 10, 10], None),
        ]
        candidate_embeddings = [[1.0, 0.0], [0.96, 0.28], [0.0, 1.0]]
        for seed in range(5):
            np.random.seed(seed)
            kept_pairs, kept_embeddings = (
                _greedy_filter_candidates_by_clip_similarity(
                    candidate_pairs,
                    candidate_embeddings,
                    similarity_threshold=0.5,
                )
            )
            boxes = [p[0] for p in kept_pairs]
            self.assertEqual(len(boxes), 2)
            self.assertIn([40, 40, 10, 10], boxes)
            self.assertEqual(len(kept_embeddings), 2)

    def test_existing_embeddings_block_duplicates(self):
        kept_pairs, _ = _greedy_filter_candidates_by_clip_similarity(
            [([0, 0, 10, 10], None)],
            [[1.0, 0.0]],
            similarity_threshold=0.5,
            existing_embeddings=[[0.96, 0.28]],
        )
        self.assertEqual(kept_pairs, [])


class TestSlidingWindowBboxes(unittest.TestCase):
    """Sliding-window bbox generation ([x, y, w, h])."""

    def _assert_inside(self, bboxes, image_size):
        for x, y, w, h in bboxes:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, image_size[0])
            self.assertLessEqual(y + h, image_size[1])

    def test_bboxes_inside_image_and_unique(self):
        image_size = (400, 400)
        bboxes = _sliding_window_bboxes_for_image(
            image_size, window_size_ratio_range=(0.15, 0.55), stride_ratio=0.3
        )
        self.assertGreater(len(bboxes), 0)
        self.assertEqual(len({tuple(b) for b in bboxes}), len(bboxes))
        self._assert_inside(bboxes, image_size)

    def test_window_size_from_ratio_range(self):
        image_size = (800, 600)
        area = image_size[0] * image_size[1]
        bboxes = _sliding_window_bboxes_for_image(
            image_size, window_size_ratio_range=(0.15, 0.55), stride_ratio=0.3
        )
        # square windows with the mid-range area
        for _, _, w, h in bboxes:
            self.assertEqual(w, h)
            self.assertAlmostEqual(w * h / area, 0.35, delta=0.01)

    def test_fixed_window_size(self):
        image_size = (500, 300)
        bboxes = _sliding_window_bboxes_for_image(
            image_size, window_size=160, stride_ratio=0.5
        )
        self.assertTrue(all(w == 160 and h == 160 for _, _, w, h in bboxes))
        self._assert_inside(bboxes, image_size)

    def test_edge_window_only_for_large_leftover_strip(self):
        # window 160, stride 80: the grid ends at x = 480. A leftover strip of
        # at most 25% of the window (40 px) is left uncovered by design;
        # a wider one gets an extra window flush with the image edge.
        narrow = _sliding_window_bboxes_for_image(
            (500, 160), window_size=160, stride_ratio=0.5
        )
        self.assertEqual(max(x + w for x, _, w, _ in narrow), 480)
        wide = _sliding_window_bboxes_for_image(
            (540, 160), window_size=160, stride_ratio=0.5
        )
        self.assertEqual(max(x + w for x, _, w, _ in wide), 540)

    def test_degenerate_image_returns_whole_image(self):
        self.assertEqual(
            _sliding_window_bboxes_for_image((1, 1)), [[0, 0, 1, 1]]
        )


class TestSubtractMask(unittest.TestCase):
    """Mask subtraction across resolutions."""

    def test_same_shape(self):
        base = np.ones((4, 4), dtype=bool)
        sub = np.zeros((4, 4), dtype=bool)
        sub[0, 0] = True
        self.assertEqual(int(_subtract_mask(base, sub).sum()), 15)

    def test_other_shape_is_resized_first(self):
        base = np.ones((4, 4), dtype=bool)
        sub = np.zeros((2, 2), dtype=bool)
        sub[0, 0] = True  # the top-left quarter at 4x4
        out = _subtract_mask(base, sub)
        self.assertEqual(out.shape, (4, 4))
        self.assertFalse(out[:2, :2].any())
        self.assertEqual(int(out.sum()), 12)

    def test_missing_mask_keeps_base(self):
        base = np.ones((2, 2), dtype=bool)
        self.assertIs(_subtract_mask(base, None), base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
