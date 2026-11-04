"""Qwen2.5-VL wrapper (the shared code is in qwen_vl)."""

from typing import Any

from transformers import AutoModelForImageTextToText

from tgcl.models.qwen_vl import QwenVLBase

__all__ = ["Qwen2_5VL"]


class Qwen2_5VL(QwenVLBase):  # pylint: disable=invalid-name  (public model name)
    """Qwen2.5-VL (e.g. Qwen/Qwen2.5-VL-7B-Instruct); everything but the model
    class is in QwenVLBase."""

    def model_class(self) -> Any:
        """AutoModelForImageTextToText (resolves to Qwen2.5-VL)."""
        return AutoModelForImageTextToText
