#!/usr/bin/env python3
"""Phrases from grounded words for the concept-word alignment BERTScore (Parekh
et al., App. C.2).

For every word set in the given word_alignment_words.json files (grounded and
random words written by tgcl/evaluation/word_alignment_eval.py),
LLaMA-3-8B-Instruct is asked for three distinct phrases that each use the
words; the numbered answer is parsed. Results go to --out ({"w1, w2, ...":
[phrase1, phrase2, phrase3]}); word sets already in --out are skipped. The model
is gated: the HF_TOKEN account must have accepted Meta's license.
"""

import argparse
import json
import os
from pathlib import Path
import re
from typing import Callable, List

import torch

from tgcl.models.revisions import revision

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
N_PHRASES = 3


def prompt_for(words: List[str]) -> str:
    """The paper's instruction (App. C.2, Table 8)."""
    return (
        f"Generate {N_PHRASES} distinct phrases, each incorporating the "
        f"words {', '.join(words)}. "
        "Ensure each phrase is clear and contextually meaningful. "
        "Answer with the numbered phrases only, one per line."
    )


def parse_phrases(text: str) -> List[str]:
    """The numbered phrases of an answer ('1. ...', '2) ...'), unquoted."""
    phrases = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+\s*[.):-]\s*(.+)", line)
        if m:
            phrases.append(m.group(1).strip().strip('"').strip())
    return phrases[:N_PHRASES]


def make_generator(device: str) -> Callable[[str], str]:
    """A function prompt -> greedy LLaMA-3 answer (model loaded on device)."""
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, token=os.environ.get("HF_TOKEN"), revision=revision(MODEL)
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
            revision=revision(MODEL),
        )
        .to(device)
        .eval()
    )

    def generate(prompt: str) -> str:
        """The model's answer to one user prompt (up to 160 new tokens)."""
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=160,
                do_sample=False,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ],
            )
        return tokenizer.decode(
            out[0, ids.shape[1] :], skip_special_tokens=True
        )

    return generate


def build_phrases(
    word_sets: List[List[str]], generate: Callable[[str], str], known: dict
) -> dict:
    """Add the phrases of every new word set to known.

    Args:
        word_sets: Word lists to make phrases from.
        generate: prompt -> answer (make_generator).
        known: {"w1, w2, ...": phrases} already made; updated in place.

    Returns:
        known. An answer with fewer than N_PHRASES phrases is retried once.
    """
    for words in word_sets:
        key = ", ".join(words)
        if not words or key in known:
            continue
        phrases = parse_phrases(generate(prompt_for(words)))
        if len(phrases) < N_PHRASES:  # one retry; keep what parses
            phrases = parse_phrases(generate(prompt_for(words))) or phrases
        known[key] = phrases
    return known


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--words",
        nargs="+",
        type=Path,
        required=True,
        help="word_alignment_words.json files",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    args = ap.parse_args()
    sets = []
    for path in args.words:
        data = json.load(open(path, encoding="utf-8"))
        sets += data["grounded"] + data["random"]
    known = (
        json.load(open(args.out, encoding="utf-8")) if args.out.exists() else {}
    )
    known = build_phrases(sets, make_generator(args.device), known)
    args.out.write_text(json.dumps(known, indent=1))
    print(f"{len(known)} word sets with phrases -> {args.out}")


if __name__ == "__main__":
    main()
