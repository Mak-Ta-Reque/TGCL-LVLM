"""Common interface of the VLM wrappers (Gemma-3n, Qwen2-VL, Qwen2.5-VL)."""

from typing import Any, Callable, Dict, Optional

from PIL import Image
import requests
import torch

__all__ = ["ImageTextModel"]


class ImageTextModel:
    """A vision-language model with its processor and prompt formatting.

    Subclasses load the model (set_model), the processor (set_processor) and
    the function that turns an instruction and an image into model inputs
    (set_preprocessor). The rest of the pipeline uses only this interface:
    get_model, get_tokenizer, get_lm_head, preprocessor, ...

    Args:
        model_name_or_path: Hugging Face model id or local path.
        processor_name: Processor id (default: the model's).
        local_files_only: Do not download from the Hub.
        device_config: tgcl.device_utils.DeviceConfig for placement (None: the
            caller moves the model).
    """

    def __init__(
        self,
        model_name_or_path: str = "google/gemma-3n-E4B-it",
        processor_name: str = "google/gemma-3n-E4B-it",
        local_files_only: bool = False,
        device_config: Optional[Any] = None,
        **kwargs: Any,
    ):

        del kwargs  # Unused: the shared wrapper interface.
        self.model_name_or_path = model_name_or_path

        self.processor_name = processor_name
        self.local_files_only = local_files_only

        # Store device_config from device_utils.DeviceConfig (or None for
        # legacy)
        self.device_config = device_config

        if processor_name is None:
            self.processor_name = model_name_or_path

        self.set_model()
        self.set_processor()
        self.set_preprocessor()

    def set_model(
        self,
    ) -> None:
        raise NotImplementedError(
            "set_model() is not defined for the model: "
            f"{self.model_name_or_path}"
        )

    def set_processor(
        self,
    ) -> None:
        raise NotImplementedError(
            "set_processor() is not defined for the model: "
            f"{self.processor_name}"
        )

    def set_preprocessor(
        self,
    ) -> None:
        raise NotImplementedError(
            "set_preprocessor() is not defined for the model: "
            f"{self.processor_name}"
        )

    def preprocess_text(
        self,
        instruction: str = "What are these?",
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError(
            "preprocess_text() is not defined for the model: "
            f"{self.model_name_or_path}"
        )

    def preprocess_images(
        self,
        image_file: str,
        **kwargs: Any,
    ):
        """Open an image from a path or an http(s) URL."""
        del kwargs  # Unused: the shared wrapper interface.
        if "http" in image_file:
            image = Image.open(
                requests.get(image_file, stream=True, timeout=30).raw
            )
        else:
            image = Image.open(image_file)
        return image

    def preprocess_input(
        self,
        instruction: str = "What are these?",
        image_file: Optional[str] = None,
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Processor inputs (pt tensors) for one image and instruction.

        Args:
            instruction: The user prompt.
            image_file: Path or URL of the image.
            response: Assistant text appended after the prompt (teacher
                forcing).
            generation_mode: Leave the assistant turn open for generation.
            **kwargs: Unused here; subclasses take more options.
        """
        del kwargs  # Unused: the shared wrapper interface.
        image = self.preprocess_images(image_file)
        text = self.preprocess_text(
            instruction=instruction,
            response=response,
            generation_mode=generation_mode,
        )

        inputs = self.processor_(images=image, text=text, return_tensors="pt")

        return inputs

    def preprocessor(
        self,
        instruction: str = "What are these?",
        image_file: str = "",
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ):
        """preprocess_input on the model's device and dtype (same arguments)."""
        del kwargs  # Unused: the shared wrapper interface.
        preprocessor = self.get_preprocessor()
        inputs = (
            preprocessor(
                instruction=instruction,
                image_file=image_file,
                response=response,
                generation_mode=generation_mode,
            )
            .to(self.get_model().device)
            .to(self.get_model().dtype)
        )

        return inputs

    def get_model(
        self,
    ) -> torch.nn.Module:

        return self.model_

    def get_language_model(
        self,
    ) -> torch.nn.Module:

        return self.model_.language_model

    def get_lm_head(
        self,
    ) -> torch.nn.Module:

        return self.model_.language_model.lm_head

    def get_processor(
        self,
    ) -> Any:

        return self.processor_

    def get_preprocessor(
        self,
    ) -> Callable:

        return self.preprocessor_

    def get_tokenizer(
        self,
    ) -> Any:

        return self.tokenizer_
