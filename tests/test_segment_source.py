"""helpers.segment_view.segment_source: the image a segment is cut from
(SEGMENT_RESOLUTION)."""

from PIL import Image

from tgcl.helpers import segment_view as sv


def test_frame_resolution_is_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SEGMENT_RESOLUTION", raising=False)
    path = tmp_path / "a.png"
    Image.new("RGB", (1280, 960)).save(path)
    img, scale = sv.segment_source(str(path), [512, 384])
    assert img.size == (512, 384) and scale == 1.0


def test_original_resolution_scales_frame_coordinates(tmp_path, monkeypatch):
    monkeypatch.setenv("SEGMENT_RESOLUTION", "original")
    path = tmp_path / "b.png"
    Image.new("RGB", (1000, 750)).save(path)
    img, scale = sv.segment_source(str(path), [512, 384])
    assert img.size == (1000, 750) and abs(scale - 1000 / 512) < 1e-9


def test_original_resolution_caps_huge_images(tmp_path, monkeypatch):
    monkeypatch.setenv("SEGMENT_RESOLUTION", "original")
    path = tmp_path / "c.png"
    Image.new("RGB", (4096, 3072)).save(path)
    img, scale = sv.segment_source(str(path), [512, 384])
    assert (
        max(img.size) == sv.MAX_SEGMENT_SOURCE_SIDE and abs(scale - 2.0) < 1e-9
    )
