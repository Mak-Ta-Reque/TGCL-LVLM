"""Pipeline step 3: per-tag decoder-layer selection via logit lens.

Runs after crops.json is built (step 2). For every concept tag it sweeps the
configured decoder layers on sampled tag regions (see
``helpers/layer_selection.py``) and writes the winning layer per tag to
``<save_dir>/logitlens/selected_layers.json``. Step 4 (``save_features.py``)
reads that file and hooks each tag's selected layer during extraction.

Runs as its own subprocess in ``scripts/run_full_pipeline.py`` so the VLM's GPU
memory is isolated from the other steps.
"""

import json
import os
from pathlib import Path

from tgcl.data import get_dataset_loader
from tgcl.device_utils import get_device_config
from tgcl.helpers.arguments import get_arguments
from tgcl.helpers.layer_selection import layer_selection_enabled
from tgcl.helpers.layer_selection import select_layer_for_tag
from tgcl.helpers.logger import log_args
from tgcl.helpers.logger import setup_logger
from tgcl.helpers.utils import set_seed
from tgcl.models import get_model_class

SELECTED_LAYERS_FILENAME = "selected_layers.json"


def selected_layers_path(save_dir: str) -> Path:
    return Path(save_dir) / "logitlens" / SELECTED_LAYERS_FILENAME


def main():
    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, "logs.log"))
    set_seed(args.seed)
    log_args(args, logger)

    if not layer_selection_enabled():
        logger.info(
            "LOGIT_LENS_LAYER_SELECTION is not enabled; skipping layer "
            "selection."
        )
        return

    assert args.dataset_name == "json_crop_map", (
        "select_layers.py expects --dataset_name json_crop_map, got "
        f"{args.dataset_name}"
    )

    device_config = get_device_config(args.device)
    device = device_config.primary_device
    logger.info(
        "Device config: %s -> primary=%s, gpu_ids=%s",
        device_config.raw,
        device,
        device_config.gpu_ids,
    )

    logger.info("Loading model: %s", args.model_name_or_path)
    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
        device_config=device_config,
    )

    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args
    )

    selections = {}
    for key, ld in loader.items():
        selection = select_layer_for_tag(
            model_class=model_class,
            tag=key,
            dataset=ld.dataset,
            args=args,
            logger=logger,
        )
        if selection is not None:
            selections[str(key)] = selection["summary"]
        else:
            logger.warning(
                "No layer selected for tag '%s'; feature extraction will fall "
                "back to the configured LAYER_PATH.",
                key,
            )

    out_path = selected_layers_path(args.save_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(selections, handle, indent=2, ensure_ascii=False)
    logger.info(
        "Saved selected layers for %s tags to %s", len(selections), out_path
    )


if __name__ == "__main__":
    main()
