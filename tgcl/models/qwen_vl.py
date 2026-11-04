"""Qwen2-VL and Qwen2.5-VL: the same chat template, vision preprocessing and
module layout (hidden states after model.language_model.norm); only the model
class differs.
"""

import os
from typing import Any, Dict, List, Optional

import torch
import transformers
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration

from tgcl.models.image_text_model import ImageTextModel
from tgcl.models.revisions import revision

__all__ = ["QwenVLBase", "QwenVL", "MIN_PIXELS", "MAX_PIXELS"]

# Qwen-VL turns every 28x28 pixel block into one visual token at the image's own
# size (dynamic
# resolution), clamped to [MIN_PIXELS, MAX_PIXELS]. Without a cap a large image
# (ImageNet val has
# 4000x3000 photos) becomes thousands of tokens: slow and out of memory. Without
# a sensible floor a
# small crop becomes a handful of tokens (a 38 px segment: 4). Gemma-3n always
# uses 256 tokens
# (768 x 768); QWEN_MIN_PIXELS=200704 gives Qwen the same budget for small
# inputs.
MAX_PIXELS = int(os.environ.get("QWEN_MAX_PIXELS", 512 * 28 * 28))
MIN_PIXELS = int(os.environ.get("QWEN_MIN_PIXELS", 4 * 28 * 28))

# transformers <= 4.55 decodes left-padded batches wrongly with the KV cache:
# padded rows stop after
# one token (rotary positions of the cached steps). Fixed in 4.57; both Qwen
# generations share the code.
MIN_TRANSFORMERS = (4, 57)


def _check_transformers() -> None:
    """Raise RuntimeError if transformers is older than MIN_TRANSFORMERS."""
    version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if version < MIN_TRANSFORMERS:
        raise RuntimeError(
            "Qwen-VL needs transformers >= "
            f"{'.'.join(map(str, MIN_TRANSFORMERS))} (found "
            f"{transformers.__version__}): older versions break batched "
            "generation. "
            "pip install -r requirements-qwen.txt"
        )


class QwenVLBase(ImageTextModel):
    """Shared Qwen-VL wrapper; subclasses name the transformers model class."""

    def model_class(self) -> Any:
        """The transformers class to load (set by each subclass)."""
        raise NotImplementedError

    def set_model(self) -> None:
        """Load the model in bfloat16; on several GPUs with the device map.

        Raises:
            RuntimeError: transformers is older than MIN_TRANSFORMERS.
        """
        _check_transformers()
        load_kwargs = dict(
            # trained in bf16; fp16 overflows in the language model
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
            revision=revision(self.model_name_or_path),
        )
        use_device_map = (
            self.device_config is not None and self.device_config.is_multi_gpu
        )
        if use_device_map:
            if self.device_config.device_map is not None:
                load_kwargs["device_map"] = self.device_config.device_map
            if self.device_config.max_memory is not None:
                load_kwargs["max_memory"] = self.device_config.max_memory
        self.model_ = (
            self.model_class()
            .from_pretrained(self.model_name_or_path, **load_kwargs)
            .eval()
        )
        if not use_device_map and self.device_config is not None:
            self.model_ = self.model_.to(self.device_config.primary_device)

    def get_language_model(self) -> torch.nn.Module:
        return self.model_.model

    def get_lm_head(self) -> torch.nn.Module:
        return self.model_.lm_head

    def set_processor(self) -> None:
        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name,
            local_files_only=self.local_files_only,
            revision=revision(self.processor_name),
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(self) -> None:
        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self,
        instruction: str = "What are these?",
        response: str = "",
        image_file: Any = None,
    ) -> List[Dict[str, Any]]:
        """The chat for one question: the user turn (image with Qwen's pixel
        limits if given, then the instruction) and, if response is set, the
        assistant's answer. No image (None or ""): no image placeholder."""
        content = [{"type": "text", "text": instruction}]
        if image_file is not None and not (
            isinstance(image_file, str) and image_file == ""
        ):
            # no image (EXPL_INPUT_MODE=text_only): no image placeholder either
            content.insert(
                0,
                {
                    "type": "image",
                    "image": image_file,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
            )
        conversation = [{"role": "user", "content": content}]
        if response:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                }
            )
        return conversation

    def preprocess_text(
        self,
        conversation: List[Dict[str, Any]],
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        """The chat as prompt text (Qwen's chat template)."""
        return self.processor_.apply_chat_template(
            conversation, add_generation_prompt=generation_mode, tokenize=False
        )

    def preprocess_images(  # pylint: disable=arguments-renamed  (Qwen takes the chat)
        self, conversation: List[Dict[str, Any]], **kwargs: Any
    ):
        """(images, videos) of a chat, resized by qwen_vl_utils."""
        del kwargs  # Unused: the shared wrapper interface.
        from qwen_vl_utils import process_vision_info

        return process_vision_info(conversation)

    def preprocess_input(
        self,
        instruction: str = "What are these?",
        image_file: Optional[Any] = None,
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Processor inputs (pt tensors) for an instruction and optional image.

        Args:
            instruction: The user prompt.
            image_file: Image path, URL or PIL image; None or "" for text only.
            response: Assistant text appended after the prompt (teacher
                forcing).
            generation_mode: Leave the assistant turn open for generation.
            **kwargs: Unused (other wrappers take more options).
        """
        del kwargs  # Unused: the shared wrapper interface.
        conversation = self.get_conversation_round(
            instruction=instruction, response=response, image_file=image_file
        )
        text = self.preprocess_text(
            conversation, generation_mode=generation_mode
        )
        has_image = any(
            c.get("type") == "image" for c in conversation[0]["content"]
        )
        if not has_image:
            return self.processor_(
                text=[text], padding=True, return_tensors="pt"
            )
        images, videos = self.preprocess_images(conversation)
        return self.processor_(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )


class QwenVL(QwenVLBase):
    """Qwen2-VL (e.g. Qwen/Qwen2-VL-7B-Instruct)."""

    def model_class(self) -> Any:
        """Qwen2VLForConditionalGeneration."""
        return Qwen2VLForConditionalGeneration
