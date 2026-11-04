"""The VLM wrappers and the loader that picks one by model name."""

import argparse
import logging
from typing import Callable, Optional, Tuple, Union

import torch

from tgcl.device_utils import parse_device_config
from tgcl.models.image_text_model import ImageTextModel

__all__ = ["get_model_class"]


SUPPORTED_MODELS = [
    "google/gemma-3n-E4B-it",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


def get_model_class(
    model_name_or_path: str = "google/gemma-3n-E4B-it",
    processor_name: str = "google/gemma-3n-E4B-it",
    device: Union[torch.device, str, None] = None,
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
    device_config=None,
) -> ImageTextModel:
    """Load model class.

    Parameters
    ----------
    device_config : DeviceConfig | None
        Preferred way to specify placement.  When supplied, *device* is ignored
        and the model is loaded with ``device_map`` / ``max_memory`` from the
        config.  When ``None``, legacy ``device`` + ``.to()`` behaviour is kept.
    """
    if device_config is None and device is not None:
        device_config = parse_device_config(str(device))

    dc_kwarg = (
        dict(device_config=device_config) if device_config is not None else {}
    )

    if "Qwen2.5-VL" in model_name_or_path:
        from tgcl.models.qwen_2_5 import Qwen2_5VL

        model_class = Qwen2_5VL(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **dc_kwarg,
        )
    elif "Qwen2-VL" in model_name_or_path:
        from tgcl.models.qwen_vl import QwenVL

        model_class = QwenVL(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **dc_kwarg,
        )
    elif "gemma-3" in model_name_or_path:
        from tgcl.models.gemma3 import Gemma3nVL

        model_class = Gemma3nVL(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **dc_kwarg,
        )
    else:
        raise NotImplementedError(
            f"Got {model_name_or_path}, but only these models are "
            f"supported {SUPPORTED_MODELS}"
        )

    if logger is not None:
        logger.info(
            f"Successfully loaded {model_name_or_path}, device config: "
            f"{device_config or device}"
        )

    # Place model on device: single-GPU device_config or legacy device string
    if getattr(model_class.model_, "hf_device_map", None) is None:
        if device_config is not None and not device_config.is_multi_gpu:
            model_class.model_.to(device_config.primary_device)
        elif device_config is None and device is not None:
            model_class.model_.to(device)

    return model_class
