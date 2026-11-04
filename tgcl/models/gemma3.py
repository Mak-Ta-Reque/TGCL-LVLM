"""Gemma-3n wrapper (the main model of the paper)."""

import functools
import os
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor
from transformers import Gemma3nForConditionalGeneration

from tgcl.models.image_text_model import ImageTextModel
from tgcl.models.revisions import revision

__all__ = ["Gemma3nVL", "share_kv_without_cache"]


def share_kv_without_cache(text_model: torch.nn.Module) -> None:
    """Make Gemma-3n's forward correct with use_cache=False.

    Gemma-3n's last `num_kv_shared_layers` decoder layers reuse the keys and
    values of earlier layers, but transformers (at least 4.55) passes them only
    through the cache object. With use_cache=False there is no cache, so those
    layers fall back to their own unused K/V projections and the model outputs
    garbage (e.g. code blocks instead of the class name). The pipeline needs
    use_cache=False (every forward sees the full sequence, which the
    hidden-state
    hooks rely on), so give each such forward a throwaway cache: same full-
    sequence computation, correct shared K/V, cache discarded afterwards.
    """
    original = text_model.forward

    @functools.wraps(original)
    def forward(*args: Any, **kwargs: Any):
        if kwargs.get("past_key_values") is None and not kwargs.get(
            "use_cache"
        ):
            kwargs["use_cache"] = True
            out = original(*args, **kwargs)
            out.past_key_values = None
            return out
        return original(*args, **kwargs)

    text_model.forward = forward


class Gemma3nVL(ImageTextModel):
    """Gemma-3n (google/gemma-3n-E4B-it) in bfloat16, eval mode."""

    def set_model(self) -> None:
        """Load the model; on several GPUs with the device_config's map."""
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
            token=os.getenv("HF_TOKEN", None),
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
        self.model_ = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **load_kwargs
        ).eval()
        if getattr(
            self.model_.config.get_text_config(), "num_kv_shared_layers", 0
        ):
            share_kv_without_cache(self.model_.model.language_model)
        if not use_device_map and self.device_config is not None:
            self.model_ = self.model_.to(self.device_config.primary_device)

    def get_language_model(self) -> torch.nn.Module:
        return (
            self.model_.model if hasattr(self.model_, "model") else self.model_
        )

    def get_lm_head(self) -> torch.nn.Module:
        return getattr(self.model_, "lm_head", None)

    def set_processor(self) -> None:
        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name,
            local_files_only=self.local_files_only,
            token=os.getenv("HF_TOKEN", None),
            revision=revision(self.processor_name),
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(self) -> None:
        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self,
        instruction: str = "Describe this image.",
        response: str = "",
        image_file: Any = None,
    ) -> List[Dict[str, Any]]:
        """The chat for one question: the user turn (image if given, then the
        instruction) and, if response is set, the assistant's answer."""
        content = [{"type": "image", "image": image_file}] if image_file else []
        content.append({"type": "text", "text": instruction})
        conversation = [{"role": "user", "content": content}]
        if response:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                }
            )
        return conversation

    def preprocess_input(
        self,
        instruction: str = "Describe this image.",
        image_file: Any = None,
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Tokenized chat (pt tensors) for an instruction and optional image.

        Args:
            instruction: The user prompt.
            image_file: Image path, URL or PIL image; None for text only.
            response: Assistant text appended after the prompt (teacher
                forcing).
            generation_mode: Leave the assistant turn open for generation.
            **kwargs: Unused (other wrappers take more options).
        """
        del kwargs  # Unused: the shared wrapper interface.
        conversation = self.get_conversation_round(
            instruction=instruction, response=response, image_file=image_file
        )
        return self.processor_.apply_chat_template(
            conversation,
            add_generation_prompt=generation_mode,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
