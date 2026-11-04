"""Step-4 datasets: the crops of each concept with the question to ask."""

import hashlib
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from tgcl.models.constants import CONCEPT_PROMPT_TEMPLATES
from tgcl.models.constants import TASK_PROMPTS

__all__ = ["JSONDataset", "select_segments"]


def _segment_area(item: Dict[str, Any]) -> float:
    bbox = item.get("bbox")
    return (
        float(bbox[2]) * float(bbox[3])
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
        else 0.0
    )


def select_segments(
    items: List[Dict[str, Any]], n: int, rng: np.random.Generator
) -> List[Dict[str, Any]]:
    """n of a class's segments for step 4 (BAG_SIZE).

    SEGMENT_SELECTION=random (default): a random subset, as in the published
    runs.
    SEGMENT_SELECTION=largest: the n largest by box area (ties in input order).
    Segments whose long side is under ~64 px of the 512-px frame are almost
    never answered with the class (ImageNet-5 SAM, Qwen2.5-VL: 0-1 %; 15-29 %
    above 256 px), so the largest ones give 1.5-2.3x the positive answers at
    the same cost.

    The random subset depends on Python's global RNG (the shuffle) as well as
    rng: step 4 seeds the global RNG (set_seed) at its start, then loads the
    model and builds the dataset, so the subset is reproducible with the same
    code and locked libraries and equal to the published runs' (a private RNG
    here would change which segments are chosen).
    """
    if len(items) <= n:
        return list(items)
    if os.environ.get("SEGMENT_SELECTION", "random") == "largest":
        order = sorted(
            range(len(items)), key=lambda i: -_segment_area(items[i])
        )
        return [items[i] for i in order[:n]]
    items = list(items)
    random.shuffle(items)
    chosen = rng.choice(items, size=n, replace=False)
    return chosen.tolist() if hasattr(chosen, "tolist") else list(chosen)


class ImageTextDataset(Dataset):
    """Base of the datasets: settings, prompt application, indexing.

    Args:
        annotation_file: Annotation file (absolute, or relative to data_dir).
        data_dir: Root of the image paths in the annotations.
        split: Data split name (kept for the subclasses).
        dataset_size: Samples kept per concept (-1: all).
        seed: Seed of the segment sampling and of the shuffled prompts.
        dataset_name: Name used in messages.
        mode: "train" appends the response to the instruction; otherwise the
            instruction alone is the text.
        prompt_template: Key of tgcl.models.constants.TASK_PROMPTS.
        shuffle_concept_prompt: P_bin_shuf ablation: name another concept in
            the prompt (see JSONDataset).
        shuffle_concept_vocab: Concepts to draw the shuffled one from.
    """

    def __init__(
        self,
        annotation_file: str = "dataset_coco.json",
        data_dir: str = "data",
        split: str = "train",
        dataset_size: int = -1,
        seed: int = 0,
        dataset_name: str = "coco",
        mode: str = "train",
        prompt_template: str = "llava",
        shuffle_concept_prompt: bool = False,
        shuffle_concept_vocab: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        del kwargs  # Unused: the dataset factory passes shared options.
        super().__init__()

        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.dataset_size = dataset_size
        self.seed = seed
        self.dataset_name = dataset_name
        self.mode = mode
        self.prompt_template = prompt_template
        # Rebuttal ablation's P_bin_shuf condition: substitute a different,
        # reproducibly-sampled concept into the prompt text (excluding the
        # crop's true tag(s)), while keeping it filed under its real tag for
        # concept-bank bucketing. See JSONDataset.create_dataset().
        self.shuffle_concept_prompt = shuffle_concept_prompt
        self.shuffle_concept_vocab = shuffle_concept_vocab

        self.rng = np.random.default_rng(seed)

        self.split = split

        self.create_dataset()

    def create_dataset(
        self,
    ) -> None:
        raise NotImplementedError(
            f"create_dataset() is not defined from dataset: {self.dataset_name}"
        )

    def apply_prompt(self, item: Dict[str, Any], mode: str = "train") -> str:
        if "train" in mode:
            text = f"{item['instruction']}{item['response']}"
        else:
            text = item["instruction"]
        return text

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        item = self.data[idx]
        item["text"] = self.apply_prompt(item, mode=self.mode)

        return item

    def __len__(
        self,
    ) -> int:
        if isinstance(self.data, dict):
            return sum(len(v) for v in self.data.values())
        return len(self.data)


class JSONDataset(ImageTextDataset):
    """The step-2 crops (crops.json) of every concept, as model inputs.

    The annotation file maps each concept (tag) to its images, and each image to
    its segments ("masks_rle": RLE mask and/or box, is_concept). Every segment
    becomes one sample: the image path, the segment, and the step-4 question
    of the prompt template with [concept] replaced by the tag. Samples are
    grouped by concept (self.data: tag -> list of samples).
    """

    _TEMPLATES = CONCEPT_PROMPT_TEMPLATES | {"llava", "non_contrastive", "null"}

    def _resolve_json_path(self) -> str:
        """The annotation file: annotation_file itself, or relative to data_dir,
        or data_dir if it is a .json file; "" if none exists."""
        if isinstance(self.annotation_file, str) and os.path.isfile(
            self.annotation_file
        ):
            return self.annotation_file
        if isinstance(self.annotation_file, str):
            candidate = os.path.join(self.data_dir, self.annotation_file)
            if os.path.isfile(candidate):
                return candidate
        if (
            isinstance(self.data_dir, str)
            and os.path.isfile(self.data_dir)
            and self.data_dir.endswith(".json")
        ):
            return self.data_dir
        return ""

    def _prompt_concept(
        self, top_key: str, rel_path: str, rel_path_to_tags: Dict[str, set]
    ) -> str:
        """The concept named in the prompt: the tag itself, or (P_bin_shuf
        ablation) a different concept drawn reproducibly per crop from
        shuffle_concept_vocab minus every tag the image is annotated under."""
        if not (self.shuffle_concept_prompt and self.shuffle_concept_vocab):
            return str(top_key)
        true_tags = rel_path_to_tags.get(rel_path, {top_key})
        candidates = [
            c for c in self.shuffle_concept_vocab if c not in true_tags
        ]
        if not candidates:
            return str(top_key)
        digest = hashlib.sha256(f"{self.seed}:{rel_path}".encode()).hexdigest()
        return random.Random(digest).choice(candidates)

    @staticmethod
    def _image_size(meta: Dict[str, Any]):
        """[width, height] from the image's metadata ([0, 0] if unreadable)."""
        image_size = meta.get("meta", {}).get("image_size", None)
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            try:
                return [int(image_size[0]), int(image_size[1])]
            except (TypeError, ValueError):
                return [0, 0]
        return image_size

    def _segment_items(
        self,
        top_key: str,
        rel_path: str,
        meta: Dict[str, Any],
        rel_path_to_tags: Dict[str, set],
    ) -> List[Dict[str, Any]]:
        """The samples of one image: one per segment with a mask or a box."""
        image_path = (
            os.path.join(self.data_dir, rel_path) if self.data_dir else rel_path
        )
        prompt_concept = self._prompt_concept(
            top_key, rel_path, rel_path_to_tags
        )
        instruction = (
            TASK_PROMPTS.get(self.prompt_template, {})
            .get("ShortCaptioning", "An image of ")
            .replace("[concept]", prompt_concept)
        )
        image_size = self._image_size(meta)
        patch_size = meta.get("meta", {}).get("patch_size", None)
        stem = os.path.splitext(os.path.basename(rel_path))[0]
        items = []
        for i, entry in enumerate(meta.get("masks_rle", None) or []):
            entry = entry if isinstance(entry, dict) else {}
            rle, bbox = entry.get("rle", None), entry.get("bbox", None)
            item = {
                "img_id": f"{stem}_mask_{i}",
                "instruction": instruction,
                "response": "",
                "image": image_path,
                "targets": "",
                "image_size": image_size,
                "patch_size": patch_size,
                "concept": top_key,
                "prompt_concept": prompt_concept,
                "is_concept": entry.get("is_concept", False),
            }
            if rle is not None:
                item["seg_mask_rle"] = rle
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    item["bbox"] = [int(v) for v in bbox]
                except (TypeError, ValueError):  # not numbers: no box
                    pass
            if rle is None and "bbox" not in item:
                continue
            items.append(item)
        return items

    def create_dataset(self) -> None:
        """Read the annotation file into self.data (tag -> samples).

        Nothing is read for prompt templates outside _TEMPLATES or when no
        annotation file exists. A tag with more than dataset_size samples keeps
        dataset_size of them (select_segments).
        """
        data = {}
        annotation_json_path = self._resolve_json_path()
        if (
            annotation_json_path.endswith(".json")
            and os.path.isfile(annotation_json_path)
            and self.prompt_template in self._TEMPLATES
        ):
            with open(annotation_json_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            if isinstance(annotations, dict) and all(
                isinstance(v, dict) for v in annotations.values()
            ):
                # An image annotated under several tags: the shuffled prompt
                # (P_bin_shuf) must exclude all of them.
                rel_path_to_tags: Dict[str, set] = {}
                if self.shuffle_concept_prompt:
                    for tag, file_map in annotations.items():
                        for rel_path in file_map:
                            rel_path_to_tags.setdefault(rel_path, set()).add(
                                tag
                            )
                for top_key, file_map in annotations.items():
                    concept_data = []
                    for rel_path, meta in file_map.items():
                        if isinstance(meta, dict):
                            concept_data += self._segment_items(
                                top_key, rel_path, meta, rel_path_to_tags
                            )
                    if (
                        self.dataset_size > 0
                        and len(concept_data) > self.dataset_size
                    ):
                        concept_data = select_segments(
                            concept_data, self.dataset_size, self.rng
                        )
                    data[top_key] = concept_data
        self.data = data
        self._rebuild_flat_index()

    # ----- Helpers for grouped datasets -----
    def _rebuild_flat_index(self) -> None:
        """
        Build a flat index mapping global idx -> (concept_key, local_idx). This
        lets the dataset behave like a standard Dataset even when data is
        grouped by concept.
        """
        self._flat_index = []
        if isinstance(self.data, dict):
            for concept_key, items in self.data.items():
                for i in range(len(items)):
                    self._flat_index.append((concept_key, i))

    def __len__(self) -> int:  # type: ignore[override]
        if isinstance(self.data, dict):
            return sum(len(v) for v in self.data.values())
        return super().__len__()

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        # Support both grouped (dict) and flat (list) storage
        if isinstance(self.data, dict):
            # Rebuild index if missing or out-of-date
            if not hasattr(self, "_flat_index") or len(self._flat_index) != sum(
                len(v) for v in self.data.values()
            ):
                self._rebuild_flat_index()
            concept_key, local_idx = self._flat_index[idx]
            item = dict(self.data[concept_key][local_idx])  # shallow copy
            # Keep track of concept the sample came from
            item["concept"] = concept_key
            item["text"] = self.apply_prompt(item, mode=self.mode)
            return item
        # Fallback to base implementation for non-dict storage
        return super().__getitem__(idx)

    def per_concept_datasets(self) -> Dict[str, Dataset]:
        """
        Return a dict mapping concept_key -> torch Dataset containing only that
        concept's items. Each item is augmented with a `text` field using the
        dataset's prompt mode.
        """
        if not isinstance(self.data, dict):
            # If data isn't grouped, treat the whole thing as a single concept
            return {
                "all": _ListDictDataset(  # type: ignore[arg-type]
                    self.data, mode=self.mode, apply_prompt_fn=self.apply_prompt
                )
            }

        per_ds: Dict[str, Dataset] = {}
        for concept_key, items in self.data.items():
            per_ds[concept_key] = _ListDictDataset(
                items,
                mode=self.mode,
                apply_prompt_fn=self.apply_prompt,
                concept_key=concept_key,
            )
        return per_ds


class _ListDictDataset(Dataset):
    """Simple wrapper dataset over a list of item dicts.

    Adds a `text` field using the provided apply_prompt_fn and optionally
    attaches `concept`.
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        mode: str,
        apply_prompt_fn: Callable[[Dict[str, Any], str], str],
        concept_key: Optional[str] = None,
    ) -> None:
        self.items = items
        self.mode = mode
        self.apply_prompt_fn = apply_prompt_fn
        self.concept_key = concept_key

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.items[idx])  # shallow copy
        if self.concept_key is not None:
            item["concept"] = self.concept_key
        item["text"] = self.apply_prompt_fn(item, mode=self.mode)
        return item
