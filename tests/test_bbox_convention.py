"""crops.json boxes are [x, y, w, h] in the image resized to IMAGE_SIZE_WIDTH;
every reader must agree."""

from PIL import Image

from tgcl.helpers.grounding import bbox_xywh_to_xyxy
from tgcl.helpers.grounding import grounding_crop
from tgcl.helpers.grounding import resize_to_width
from tgcl.preprocessing.crops_to_json import _compute_virtual_resize
from tgcl.preprocessing.crops_to_json import _sliding_window_bboxes_for_image


def test_sliding_windows_keep_their_size():
    # Before the fix, [192, 0, 160, 160] was read as corners: a 1-pixel strip.
    boxes = _sliding_window_bboxes_for_image((512, 384), window_size=160)
    assert len(boxes) > 10
    for box in boxes:
        x1, y1, x2, y2 = bbox_xywh_to_xyxy(box, 512, 384)
        assert (x1, y1) == tuple(box[:2])
        assert (x2 - x1, y2 - y1) == (160, 160)


def test_clipped_to_image_and_never_empty():
    assert bbox_xywh_to_xyxy([480, 360, 160, 160], 512, 384) == (
        480,
        360,
        512,
        384,
    )
    assert bbox_xywh_to_xyxy([-5, 600, 0, 0], 512, 384) == (0, 383, 1, 384)


def test_resize_matches_crops_to_json():
    for size in [(640, 480), (480, 640), (333, 500), (1024, 77)]:
        assert (
            resize_to_width(Image.new("RGB", size), 512).size
            == _compute_virtual_resize(*size, 512)[:2]
        )


def test_grounding_crop_is_the_step4_window(tmp_path):
    # 640x480 original; the box is in the 512x384 image crops_to_json worked on.
    img = Image.new("RGB", (640, 480), (0, 0, 0))
    img.paste(
        (255, 0, 0), (400, 100, 600, 300)
    )  # red square, original coordinates
    img.save(tmp_path / "x.png")
    crop = grounding_crop(
        tmp_path / "x.png", [320, 80, 160, 160], 512
    )  # same square at 0.8 scale
    assert crop.size == (160, 160)
    r, g, b = crop.resize((1, 1)).getpixel((0, 0))
    assert r > 240 and g < 15 and b < 15


def test_scorer_uses_the_same_crop(tmp_path):
    import tgcl.evaluation.clip_bert_score_eval as scorer

    Image.new("RGB", (640, 480)).save(tmp_path / "y.png")
    assert scorer._grounding_crop(
        str(tmp_path / "y.png"), [192, 0, 160, 160]
    ).size == (160, 160)
