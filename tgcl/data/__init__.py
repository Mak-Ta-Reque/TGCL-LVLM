"""Step-4 input: the crops of every concept as DataLoaders."""

import argparse
import logging
from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import DataLoader

from tgcl.data.image_text_dataset import JSONDataset

__all__ = ["get_dataset_loader"]


def safe_collate_dicts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of dicts into a dict of lists, sanitizing None values.

    - For strings: replace None with ""
    - For lists: replace None with []
    - For numbers/tensors: replace None with 0 / empty tensor is not handled
      here (not needed now)
    - For other types: replace None with ""
    """
    if not batch:
        return {}
    out: Dict[str, Any] = {}
    keys = set()
    for d in batch:
        keys.update(d.keys())
    for k in keys:
        vals = [d.get(k, None) for d in batch]
        # Infer a simple target type
        sample = next((v for v in vals if v is not None), None)
        if sample is None:
            # Every value in the batch is None for this key (e.g. patch_size
            # when no cropping is applied) — leave as None rather than
            # coercing to "", which breaks numeric parsing downstream
            # (callers already handle None explicitly).
            out[k] = vals
        elif isinstance(sample, str):
            out[k] = [v if v is not None else "" for v in vals]
        elif isinstance(sample, list):
            out[k] = [v if v is not None else [] for v in vals]
        else:
            out[k] = [v if v is not None else "" for v in vals]
    return out


def get_dataset_loader(
    dataset_name: str,
    args: Optional[argparse.Namespace] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Dict[str, DataLoader]:
    """One DataLoader per concept of the step-2 crops (json_crop_map).

    Args:
        dataset_name: Must be "json_crop_map" (the pipeline's only dataset).
        args: Feature-saving arguments (data_dir, annotation_file, split,
            dataset_size, seed, generation_mode, prompt_template, batch_size,
            shuffle_concept_prompt, shuffle_concept_vocab, ...).
        logger: Optional logger.
        **kwargs: Unused.

    Returns:
        Concept name -> DataLoader of its samples (batches collated by
        safe_collate_dicts, not shuffled).

    Raises:
        NotImplementedError: Another dataset name.
    """
    del kwargs  # Unused: callers may pass loader options this dataset ignores.
    if dataset_name != "json_crop_map":
        raise NotImplementedError(f"{dataset_name} is not implemented.")
    dataset = JSONDataset(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        split=args.split,
        dataset_size=args.dataset_size,
        seed=args.seed,
        dataset_name=args.dataset_name,
        mode=("val" if args.generation_mode else "train"),
        prompt_template=args.prompt_template,
        shuffle_concept_prompt=getattr(args, "shuffle_concept_prompt", False),
        shuffle_concept_vocab=(
            [
                c.strip()
                for c in args.shuffle_concept_vocab.split(",")
                if c.strip()
            ]
            if getattr(args, "shuffle_concept_vocab", None)
            else None
        ),
    )
    if logger is not None:
        logger.info(f"Successfully loaded {dataset_name}")
    dataloaders = {}
    for key, ds in dataset.per_concept_datasets().items():
        if logger:
            logger.info(f"Concept: {key}, samples: {len(ds)}")
        dataloaders[key] = DataLoader(
            ds,
            batch_size=getattr(args, "batch_size", 1),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=safe_collate_dicts,
        )
    return dataloaders
