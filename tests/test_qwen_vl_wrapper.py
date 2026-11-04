"""models.qwen_vl: Qwen2-VL and Qwen2.5-VL share one wrapper (no weights loaded
here)."""

from PIL import Image
import pytest

from tgcl.models import qwen_vl
from tgcl.models.qwen_2_5 import Qwen2_5VL
from tgcl.models.qwen_vl import QwenVL
from tgcl.models.qwen_vl import QwenVLBase


def _bare(cls):
    return object.__new__(cls)  # skip __init__: it loads weights


@pytest.mark.parametrize("cls", [QwenVL, Qwen2_5VL])
def test_image_conversation_carries_pixel_limits(cls):
    conv = _bare(cls).get_conversation_round(
        "Classify.", image_file=Image.new("RGB", (40, 30))
    )
    image, text = conv[0]["content"]
    assert image["type"] == "image" and text == {
        "type": "text",
        "text": "Classify.",
    }
    assert (image["min_pixels"], image["max_pixels"]) == (
        qwen_vl.MIN_PIXELS,
        qwen_vl.MAX_PIXELS,
    )


@pytest.mark.parametrize("cls", [QwenVL, Qwen2_5VL])
@pytest.mark.parametrize("no_image", [None, ""])
def test_text_only_has_no_image_placeholder(cls, no_image):
    conv = _bare(cls).get_conversation_round(
        "Write 'car'.", image_file=no_image
    )
    assert conv[0]["content"] == [{"type": "text", "text": "Write 'car'."}]


def test_response_becomes_assistant_turn():
    conv = _bare(QwenVL).get_conversation_round(
        "Q", response="car", image_file=None
    )
    assert conv[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "car"}],
    }


def test_model_classes():
    from transformers import Qwen2VLForConditionalGeneration

    assert _bare(QwenVL).model_class() is Qwen2VLForConditionalGeneration
    assert (
        issubclass(Qwen2_5VL, QwenVLBase)
        and _bare(Qwen2_5VL).model_class() is not None
    )
