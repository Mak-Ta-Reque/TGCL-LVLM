#!/usr/bin/env python3
"""
Dataset Inference Script for Vision Language Models

This script iterates through a dataset of images organized in subfolders or
directly in the root directory,
runs inference using Hugging Face vision language models,
and saves the results to a CSV file.

The script automatically detects if models are cached locally and uses them for
offline inference.
If models are not cached, they will be downloaded from Hugging Face Hub.

Usage:
    # Basic usage - automatically detects local cache or downloads from Hub
    python dataset_inference.py --dataset_path /path/to/dataset \\
        --model_name google/gemma-3n-E4B-it --output_csv results.csv

    # Limit to a random sample of 500 images with fixed seed
    python dataset_inference.py --dataset_path /path/to/dataset \\
        --model_name google/gemma-3n-E4B-it --output_csv results.csv \\
        --image_budget 500 --seed 42

    # With HF token for private models
    export HF_TOKEN=your_token
    python dataset_inference.py --dataset_path /path/to/dataset \\
        --model_name google/gemma-3n-E4B-it --output_csv results.csv
"""

import os

# Configure torch compilation and dynamo settings for Gemma
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# TORCH_COMPILE_DISABLE must be set before torch is imported.
# pylint: disable=wrong-import-position
import argparse
import csv
import gc
import logging
from pathlib import Path
import random
import sys
from typing import List, Optional, Tuple

from huggingface_hub import login
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import Gemma3nForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor

from tgcl.device_utils import get_device_config
from tgcl.helpers.utils import set_seed
from tgcl.models.revisions import revision

# pylint: enable=wrong-import-position

# HF_HOME should be set via .env; no hardcoded fallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _cleanup_gpu(logger_obj: Optional[logging.Logger] = None) -> None:
    """Free cached GPU memory after the step. Only touches CUDA if this process
    already uses it: a bare synchronize() opened a context on the current
    device (GPU 0), which runs out of memory while another job holds it."""
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        if logger_obj is not None:
            logger_obj.info('GPU cleanup done in dataset_inference')


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Popular Hugging Face vision language models
POPULAR_MODELS = {
    'gemma-3n': 'google/gemma-3n-E4B-it',
    'gemma-3n-e4b': 'google/gemma-3n-E4B-it',
    'qwen2-vl-7b': 'Qwen/Qwen2-VL-7B-Instruct',
    'qwen2.5-vl-7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
}


class SafeNanLogitsProcessor(LogitsProcessor):
    """Sanitize logits to avoid NaNs/Infs before sampling.

    Replaces NaN with a large negative value, clamps infinities, and bounds
    scores.
    """

    def __init__(self, neg_inf: float = -1e9, pos_inf: float = 1e9):
        super().__init__()
        self.neg_inf = neg_inf
        self.pos_inf = pos_inf

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores = torch.nan_to_num(
            scores, nan=self.neg_inf, posinf=self.pos_inf, neginf=self.neg_inf
        )
        return torch.clamp(scores, min=self.neg_inf, max=self.pos_inf)


def get_image_files(
    dataset_path: str, image_budget: Optional[int] = None, seed: int = 42
) -> List[Tuple[str, str, str]]:
    """
    Get image files from the dataset directory, including files in the root and
    in immediate subfolders. Optionally sample up to image_budget images from
    the full collected set.

    Args:
        dataset_path: Path to the dataset directory
        image_budget: If provided and >0, randomly sample up to this many images
        total
        seed: Random seed for reproducible per-subfolder sampling
    Returns:
        List of tuples containing (root_path, subfolder, image_name)
    """
    image_files: List[Tuple[str, str, str]] = []
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise ValueError(f'Dataset path does not exist: {dataset_path}')

    # Include images directly in root directory
    for image_file in dataset_path.iterdir():
        if (
            image_file.is_file()
            and image_file.suffix.lower() in IMAGE_EXTENSIONS
        ):
            image_files.append((str(dataset_path), '', image_file.name))

    rng = random.Random(seed)

    # Iterate through subdirectories and collect images
    for subfolder in dataset_path.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name
            for image_file in subfolder.iterdir():
                if (
                    image_file.is_file()
                    and image_file.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    image_files.append(
                        (str(dataset_path), subfolder_name, image_file.name)
                    )

    # Apply global sampling if requested
    if (
        image_budget is not None
        and image_budget > 0
        and len(image_files) > image_budget
    ):
        rng.shuffle(image_files)
        image_files = image_files[:image_budget]

    logger.info(
        'Collected %s images (budget=%s)', len(image_files), image_budget
    )
    return image_files


def resize_image(
    image: Image.Image, target_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: PIL Image object
        target_size: Tuple of (width, height) or None for no resizing

    Returns:
        Resized PIL Image
    """
    if target_size is None:
        return image

    # Calculate the aspect ratio
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate scaling factor to maintain aspect ratio
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    resized_image = image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS
    )

    # Create a new image with target size and paste the resized image
    final_image = Image.new('RGB', target_size, (255, 255, 255))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    final_image.paste(resized_image, (paste_x, paste_y))

    return final_image


def resize_image_by_width(
    image: Image.Image, target_width: Optional[int] = None
) -> Image.Image:
    """Resize image to a reference width and compute height from aspect ratio.

    This performs a direct resize to (target_width, target_height) with no
    padding.
    """
    if target_width is None:
        return image
    tw = int(target_width)
    if tw <= 0:
        return image
    original_width, original_height = image.size
    if original_width <= 0 or original_height <= 0:
        return image
    scale = tw / float(original_width)
    th = max(1, int(round(original_height * scale)))
    if (original_width, original_height) == (tw, th):
        return image
    return image.resize((tw, th), Image.Resampling.LANCZOS)


def load_huggingface_model(
    model_name: str,
    trust_remote_code: bool = True,
    hf_token: Optional[str] = None,
    device_str: Optional[str] = None,
) -> Tuple[object, object]:
    """
    Load Hugging Face vision language model and processor. Automatically detects
    if model exists in cache and uses local files if available.

    Args:
        model_name: Hugging Face model name or path
        trust_remote_code: Whether to trust remote code
        hf_token: Hugging Face authentication token for private models
        device_str: Device config string (e.g. 'cuda:1', 'auto', etc.)

    Returns:
        Tuple of (model, processor)
    """
    # Default HF cache ($HF_HOME/hub): cached files are reused and missing ones
    # (e.g. after an interrupted download) fetched. HF_HUB_OFFLINE=1 forbids
    # network.
    use_local_files = os.environ.get('HF_HUB_OFFLINE', '0').lower() in (
        '1',
        'true',
        'yes',
    )
    if use_local_files:
        logger.info('HF_HUB_OFFLINE set: loading from the local HF cache only')

    # Prepare common loading arguments
    loading_kwargs = {
        'trust_remote_code': trust_remote_code,
        'local_files_only': use_local_files,
        'revision': revision(model_name),
    }

    # Add token if provided
    if hf_token:
        loading_kwargs['token'] = hf_token
        # Login to Hugging Face Hub
        login(token=hf_token)
    else:
        logger.info('No HF token provided, proceeding without authentication')
    # Parse device configuration
    device_config = get_device_config(device_str)
    logger.info(
        'Device config: %s -> gpu_ids=%s',
        device_config.raw,
        device_config.gpu_ids,
    )

    # For single-GPU or CPU: load without device_map then .to(device)
    # For multi-GPU or auto: use device_map + max_memory (requires
    # accelerate)
    use_device_map = device_config.is_multi_gpu
    dm_kwargs = {}
    target_device = device_config.primary_device  # e.g. cuda:1 or cpu
    if torch.device(target_device).type == 'cuda':
        # Default-device CUDA calls must land on this job's GPU, not GPU 0.
        torch.cuda.set_device(torch.device(target_device))
    if use_device_map:
        if device_config.device_map is not None:
            dm_kwargs['device_map'] = device_config.device_map
        if device_config.max_memory is not None:
            dm_kwargs['max_memory'] = device_config.max_memory

    # Try different model classes based on model name
    if 'qwen' in model_name.lower():
        # Prefer bfloat16 to reduce overflow/NaNs when supported
        preferred_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            token=loading_kwargs['token']
            if 'token' in loading_kwargs
            else None,
            torch_dtype=preferred_dtype,
            revision=revision(model_name),
            **dm_kwargs,
        ).eval()
        if not use_device_map:
            model = model.to(target_device)
    elif 'gemma-3n' in model_name.lower() or 'gemma3n' in model_name.lower():
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            **dm_kwargs,
            **loading_kwargs,
        ).eval()
        if not use_device_map:
            model = model.to(target_device)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            **dm_kwargs,
            **loading_kwargs,
        )
        if not use_device_map:
            model = model.to(target_device)

    processor = AutoProcessor.from_pretrained(model_name, **loading_kwargs)

    # Ensure tokenizer padding/eos are set and left padding for decoder-only
    if hasattr(processor, 'tokenizer'):
        tok = processor.tokenizer
        if (
            getattr(tok, 'pad_token_id', None) is None
            and getattr(tok, 'eos_token_id', None) is not None
        ):
            tok.pad_token = tok.eos_token
        tok.padding_side = 'left'
        # Reflect into generation_config when available
        gen_cfg = getattr(model, 'generation_config', None)
        if gen_cfg is not None:
            if (
                getattr(gen_cfg, 'eos_token_id', None) is None
                and getattr(tok, 'eos_token_id', None) is not None
            ):
                gen_cfg.eos_token_id = tok.eos_token_id
            if (
                getattr(gen_cfg, 'pad_token_id', None) is None
                and getattr(tok, 'pad_token_id', None) is not None
            ):
                gen_cfg.pad_token_id = tok.pad_token_id

    # Log where the model actually lives
    if use_device_map:
        hf_map = getattr(model, 'hf_device_map', None)
        if hf_map:
            devices_used = sorted(set(str(v) for v in hf_map.values()))
            logger.info('Model loaded with device_map across: %s', devices_used)
        else:
            logger.info(
                'Model loaded with device_map (device: %s)', model.device
            )
    else:
        logger.info(
            'Model loaded on %s (model.device=%s)',
            target_device,
            model.device,
        )
    return model, processor


def process_dataset(
    dataset_path: str,
    model_name: str,
    output_csv: str,
    prompt: str = 'Describe this image.',
    image_size: Optional[Tuple[int, int]] = None,
    image_size_width: Optional[int] = None,
    trust_remote_code: bool = True,
    resume: bool = False,
    hf_token: Optional[str] = None,
    batch_size: int = 1,
    image_budget: Optional[int] = None,
    seed: int = 42,
    device_str: Optional[str] = None,
) -> None:
    """
    Process the entire dataset and save results to CSV.

    Args:
        dataset_path: Path to the dataset directory
        model_name: Hugging Face model name or path
        output_csv: Path to output CSV file
        prompt: Text prompt for the model
        image_size: Optional image resize dimensions (width, height)
        (letterboxed)
        image_size_width: Optional reference width; height is computed from
        aspect ratio
        trust_remote_code: Whether to trust remote code
        resume: Whether to resume from existing CSV file
        hf_token: Hugging Face authentication token for private models
        image_budget: If provided, randomly sample up to this many images (after
        resume filtering)
        seed: Seed of the image sample and of the answers' sampling (the
            object lists are sampled, do_sample=True).
    """
    # Without a torch seed, every run listed different objects for the same
    # images, and the concept map and all later steps differed with it.
    set_seed(seed)
    # Get all image files
    image_files = get_image_files(
        dataset_path, image_budget=image_budget, seed=seed
    )

    # Check if resuming from existing file
    processed_files = set()
    if resume and os.path.exists(output_csv):
        with open(output_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle root directory images differently
                root = row.get('root_path', '')
                rel = row.get('image_relpath')
                name = row.get('image_name', '')
                if rel:
                    processed_files.add(os.path.join(root, rel))
                else:
                    # Backward-compat for old CSV with 'subfolder'
                    sub = row.get('subfolder', '')
                    if sub:
                        processed_files.add(os.path.join(root, sub, name))
                    else:
                        processed_files.add(os.path.join(root, name))
        logger.info(
            'Resuming: %s files already processed', len(processed_files)
        )

    # Filter out already processed files if resuming
    if resume:
        image_files = [
            (root, subfolder, img)
            for root, subfolder, img in image_files
            if os.path.join(root, subfolder, img) not in processed_files
        ]
        logger.info('Remaining files to process: %s', len(image_files))

    model = None
    processor = None
    try:
        # Load the model and processor
        model, processor = load_huggingface_model(
            model_name, trust_remote_code, hf_token, device_str=device_str
        )

        # Prepare CSV file (no 'subfolder'; using 'image_relpath')
        fieldnames = [
            'root_path',
            'image_relpath',
            'image_name',
            'predicted_text',
            'prompt_used',
        ]
        mode = 'a' if resume and os.path.exists(output_csv) else 'w'

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx : min(ndx + n, l)]

        with open(output_csv, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if mode == 'w':
                writer.writeheader()

            for image_batch in tqdm(
                list(batch(image_files, batch_size)), desc='Processing batches'
            ):
                batch_images = []
                batch_prompts = []
                batch_paths = []  # (root_path, image_relpath, image_name)
                for root_path, subfolder, image_name in image_batch:
                    image_relpath = (
                        os.path.join(subfolder, image_name)
                        if subfolder
                        else image_name
                    )
                    image_path = os.path.join(root_path, image_relpath)
                    try:
                        image = Image.open(image_path).convert('RGB')
                        if image_size_width:
                            image = resize_image_by_width(
                                image, image_size_width
                            )
                        elif image_size:
                            image = resize_image(image, image_size)
                    except OSError as e:  # missing or unreadable image
                        logger.error(
                            'Error loading image %s: %s', image_path, e
                        )
                        image = None
                    batch_images.append(image)
                    batch_prompts.append(prompt)
                    batch_paths.append((root_path, image_relpath, image_name))

                # Remove None images
                valid_indices = [
                    i for i, img in enumerate(batch_images) if img is not None
                ]
                valid_images = [batch_images[i] for i in valid_indices]
                valid_prompts = [batch_prompts[i] for i in valid_indices]
                valid_paths = [batch_paths[i] for i in valid_indices]

                if not valid_images:
                    continue

                # Prepare batch inputs
                if 'qwen' in model_name.lower():
                    messages = [
                        [
                            {
                                'role': 'user',
                                'content': [
                                    {'type': 'image', 'image': img},
                                    {'type': 'text', 'text': pr},
                                ],
                            }
                        ]
                        for img, pr in zip(valid_images, valid_prompts)
                    ]
                    if hasattr(processor, 'apply_chat_template'):
                        text_prompts = [
                            processor.apply_chat_template(
                                m, tokenize=False, add_generation_prompt=True
                            )
                            for m in messages
                        ]
                        inputs = processor(
                            text=text_prompts,
                            images=valid_images,
                            return_tensors='pt',
                            padding=True,
                            padding_side='left',
                        )
                    else:
                        inputs = processor(
                            text=valid_prompts,
                            images=valid_images,
                            return_tensors='pt',
                            padding=True,
                            padding_side='left',
                        )
                elif 'gemma' in model_name.lower():
                    messages = [
                        [
                            {
                                'role': 'system',
                                'content': [
                                    {
                                        'type': 'text',
                                        'text': 'You are a helpful assistant.',
                                    }
                                ],
                            },
                            {
                                'role': 'user',
                                'content': [
                                    {'type': 'image', 'image': img},
                                    {'type': 'text', 'text': pr},
                                ],
                            },
                        ]
                        for img, pr in zip(valid_images, valid_prompts)
                    ]
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors='pt',
                    )
                else:
                    inputs = processor(
                        text=valid_prompts,
                        images=valid_images,
                        return_tensors='pt',
                        padding=True,
                        padding_side='left',
                    )

                device = next(model.parameters()).device
                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                with torch.no_grad():
                    if 'gemma' in model_name.lower():
                        input_len = inputs['input_ids'].shape[-1]
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=True,
                            pad_token_id=processor.tokenizer.eos_token_id
                            if hasattr(processor, 'tokenizer')
                            else None,
                        )
                        new_tokens = [out[input_len:] for out in outputs]
                        generated_texts = [
                            processor.decode(
                                nt, skip_special_tokens=True
                            ).strip()
                            for nt in new_tokens
                        ]
                    else:
                        logits_processor = [SafeNanLogitsProcessor()]
                        safe_temperature = max(1e-4, 0.7)
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=150,
                            do_sample=True,
                            temperature=safe_temperature,
                            logits_processor=logits_processor,
                            pad_token_id=processor.tokenizer.eos_token_id
                            if hasattr(processor, 'tokenizer')
                            else None,
                        )
                        if hasattr(processor, 'tokenizer'):
                            tokenizer = processor.tokenizer
                        else:
                            tokenizer = processor
                        if 'input_ids' in inputs:
                            generated_texts = [
                                tokenizer.decode(
                                    out[inputs['input_ids'].shape[1] :],
                                    skip_special_tokens=True,
                                ).strip()
                                for out in outputs
                            ]
                        else:
                            generated_texts = [
                                tokenizer.decode(
                                    out, skip_special_tokens=True
                                ).strip()
                                for out in outputs
                            ]

                # Write results for each image in batch
                for (
                    root_path,
                    image_relpath,
                    image_name,
                ), predicted_text in zip(valid_paths, generated_texts):
                    writer.writerow(
                        {
                            'root_path': root_path,
                            'image_relpath': image_relpath,
                            'image_name': image_name,
                            'predicted_text': predicted_text,
                            'prompt_used': prompt,
                        }
                    )
                csvfile.flush()

        logger.info('Processing complete! Results saved to %s', output_csv)
    finally:
        # Release model/processors and free CUDA memory before returning to
        # caller.
        del model, processor
        _cleanup_gpu(logger)


def main(argv=None) -> None:
    """Main function to run the dataset inference script."""
    parser = argparse.ArgumentParser(
        description='Run inference on image dataset using Hugging Face vision '
        'language models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Gemma 3n model (auto-detects local cache)
  python dataset_inference.py --dataset_path /path/to/dataset --model_name google/gemma-3n-E4B-it

  # Using Qwen2-VL with custom prompt and image resizing
  python dataset_inference.py --dataset_path /path/to/dataset --model_name Qwen/Qwen2-VL-7B-Instruct \\
    --prompt "What objects are in this image?" --image_size 512 512

    # Resize by reference width (height auto from aspect ratio)
    python dataset_inference.py --dataset_path /path/to/dataset --model_name Qwen/Qwen2-VL-7B-Instruct \\
        --prompt "What objects are in this image?" --image_size_width 512

  # Resume interrupted processing
  python dataset_inference.py --dataset_path /path/to/dataset --model_name google/gemma-3n-E4B-it --resume

  # Using private model with HF token
  python dataset_inference.py --dataset_path /path/to/dataset --model_name private/model --hf_token your_token

Popular models:
  - google/gemma-3n-E4B-it (Gemma 3n)
  - Qwen/Qwen2-VL-7B-Instruct (Qwen2-VL 7B)
  - Qwen/Qwen2.5-VL-7B-Instruct (Qwen2.5-VL 7B)
        """,
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        required=False,
        help='Path to the dataset directory containing image subfolders and/or '
        'images in the root directory',
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default=os.environ.get('VLM_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct'),
        help='Hugging Face model name or path',
    )

    parser.add_argument(
        '--output_csv',
        type=str,
        default='dataset_inference_results.csv',
        help='Output CSV file path (default: dataset_inference_results.csv)',
    )

    parser.add_argument(
        '--prompt',
        type=str,
        default='Describe this image.',
        help='Text prompt for the model (default: "Describe this image.")',
    )

    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help='Resize images to specified dimensions (width height), e.g., '
        '--image_size 512 512',
    )

    parser.add_argument(
        '--image_size_width',
        type=int,
        default=None,
        metavar='WIDTH',
        help='Resize images to WIDTH and compute HEIGHT from aspect ratio (no '
        'padding)',
    )

    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        default=True,
        help='Trust remote code when loading models (default: True)',
    )

    parser.add_argument(
        '--no_trust_remote_code',
        action='store_true',
        help='Do not trust remote code when loading models',
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing CSV file (skip already processed images)',
    )

    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List popular model names and exit',
    )

    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Hugging Face authentication token for private models (or set '
        'HF_TOKEN environment variable)',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=int(os.environ.get('BATCH_SIZE', '10')),
        help='Batch size for inference',
    )
    parser.add_argument(
        '--image_budget',
        type=int,
        default=None,
        help='Randomly sample up to this many images from the (remaining) '
        'dataset before processing',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=int(os.environ.get('SEED', '42')),
        help='Random seed for image sampling (default: 42)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=os.environ.get('DEVICE', 'auto'),
        help="Device config. Formats: 'cuda:1', 'cuda:0,2', 'cuda:[0,2]', "
        "'auto', 'cpu'.",
    )

    args = parser.parse_args(argv)

    # Handle HF token from environment variable if not provided via argument
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if hf_token:
        logger.info('Using Hugging Face authentication token')

    # Handle model listing
    if args.list_models:
        print('Popular Hugging Face Vision Language Models:')
        print('=' * 50)
        for short_name, full_name in POPULAR_MODELS.items():
            print(f'{short_name:<15} : {full_name}')
        print(
            '\nYou can use either the short name or full Hugging Face '
            'model path.'
        )
        return

    # Check if dataset_path is provided when not listing models
    if not args.dataset_path:
        print('Error: --dataset_path is required when not using --list_models')
        parser.print_help()
        return

    # Handle trust_remote_code flag
    trust_remote_code = args.trust_remote_code and not args.no_trust_remote_code

    # Convert short model names to full names if needed
    model_name = args.model_name
    if model_name in POPULAR_MODELS:
        model_name = POPULAR_MODELS[model_name]
        print(f'Using model: {model_name}')

    image_size_width = (
        int(args.image_size_width) if args.image_size_width else None
    )

    # Convert image_size to tuple if provided
    image_size = tuple(args.image_size) if args.image_size else None
    if image_size_width:
        print(
            f'Images will be resized by width={image_size_width} (height '
            'auto from aspect ratio)'
        )
    elif image_size:
        print(f'Images will be resized to: {image_size[0]}x{image_size[1]}')

    # Run the processing
    try:
        process_dataset(
            dataset_path=args.dataset_path,
            model_name=model_name,
            output_csv=args.output_csv,
            prompt=args.prompt,
            image_size=image_size,
            image_size_width=image_size_width,
            batch_size=args.batch_size,
            trust_remote_code=trust_remote_code,
            resume=args.resume,
            hf_token=hf_token,
            image_budget=args.image_budget,
            seed=args.seed,
            device_str=args.device,
        )
    except KeyboardInterrupt:
        print('\n⚠️ Processing interrupted by user')
        sys.exit(130)


if __name__ == '__main__':
    main()
