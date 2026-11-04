"""Batching of per-sample VLM processor outputs (pipeline steps 4 and 6)."""

from typing import Any, Dict, List

import torch


def collate_inputs(
    per_sample_inputs: List[Dict[str, Any]], pad_id: int = 0
) -> Dict[str, Any]:
    """Collate per-sample processor outputs into a batch dict.

    Any (1, seq_len) tensor that is parallel to that sample's ``input_ids``
    (attention_mask, and e.g. Qwen's ``mm_token_type_ids`` on newer
    transformers) is left-padded together with ``input_ids`` — sequence lengths
    differ per sample whenever image sizes differ, and a plain ``torch.cat``
    would fail or silently misalign them. Everything else is concatenated along
    dim 0 (pixel_values patches, image_grid_thw, ...).
    """
    if len(per_sample_inputs) == 1:
        return per_sample_inputs[0]
    batched = {}
    keys = per_sample_inputs[0].keys()

    seq_lens = [
        int(d["input_ids"].shape[-1]) if "input_ids" in d else None
        for d in per_sample_inputs
    ]

    def _is_token_aligned(vals: List[Any]) -> bool:
        return all(
            isinstance(v, torch.Tensor)
            and v.ndim == 2
            and v.shape[0] == 1
            and seq_lens[i] is not None
            and v.shape[1] == seq_lens[i]
            for i, v in enumerate(vals)
        )

    for k in keys:
        vals = [d[k] for d in per_sample_inputs]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            if k in ("input_ids", "attention_mask") or _is_token_aligned(vals):
                seqs = []
                for v in vals:
                    if v.ndim == 2 and v.shape[0] == 1:
                        seqs.append(v.squeeze(0))
                    else:
                        seqs.append(v)
                padding_value = pad_id if k == "input_ids" else 0
                batched[k] = torch.nn.utils.rnn.pad_sequence(
                    seqs,
                    batch_first=True,
                    padding_side="left",
                    padding_value=padding_value,
                )
            else:
                arrs = []
                for v in vals:
                    if v.ndim == 0:
                        v = v.unsqueeze(0)
                    arrs.append(v)
                try:
                    batched[k] = torch.cat(arrs, dim=0)
                except RuntimeError:  # shapes differ: keep the list
                    batched[k] = vals
        else:
            batched[k] = vals
    return batched
