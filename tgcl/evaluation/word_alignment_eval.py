#!/usr/bin/env python3
"""Concept-word alignment: how well the words grounding a test image's top
concepts describe that image, against random words. Evaluation protocol of
CoX-LMM (Parekh et al., NeurIPS 2024; arXiv 2406.08074, Sec. 4.1, App. C.2),
applied to any concept bank -- not the CoX-LMM method itself
(configs/paper/coxlmm.env).

For every explained test image:
  representation  the real hidden state of the answer's first scored token (the
                  token of interest; vlm_explanations_activations.npy)
  activations     the decomposition's transform of that representation onto the
                  concept bank (snmf: positive lasso sparse code, as
                  sklearn DictionaryLearning.transform; pca: projection; simple/
                  simple: 1/(1+distance)); top-k concepts by activation
  word_clipscore  test image vs "A photo is described by the words w1, w2, ..."
                  built from the concept's grounded words (<= 10 English,
                  non-stop,
                  >= 3-letter words), CLIP ViT-B/32, 2.5 * max(cos, 0)
                  (src/metrics/clipscore.py preprocessing and caption prefix)
  word_bertscore  the concept's words merged into one text ("w1, w2, ...") vs
  the image's
                  ground-truth captions (--captions_json): BERTScore F1, max
                  over captions
  phrase_bertscore  max over the concept's LLaMA-3 phrases (--phrases_json, from
                  eval/word_alignment_phrases.py) x the image's ground-truth
                  captions
                  (--captions_json), RoBERTa-large, no rescaling; needs both
                  files
  random_word_clipscore / random_word_bertscore / random_phrase_bertscore
                  the same scores with each concept's words replaced by words
                  decoded from a random direction through the LM head (same
                  filters, same count), saved to <out>/word_alignment_words.json
top-k = mean over the top-k concepts' scores (k=1 is the paper's reported
value).

Writes <out>/word_alignment_table.csv and <out>/word_alignment_per_sample.csv.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np
from sklearn.decomposition import sparse_encode
import torch

from tgcl.helpers.image_refs import is_memory_ref
from tgcl.helpers.image_refs import source_image
from tgcl.helpers.nltk_data import english_words
from tgcl.helpers.utils import apply_determinism_setting
from tgcl.metrics.clipscore import CLIPCaptionDataset
from tgcl.metrics.clipscore import CLIPImageDataset
from tgcl.metrics.utils import get_stopwords
from tgcl.metrics.utils import GIST_FILE_PATH
from tgcl.metrics.utils import is_number
from tgcl.models.revisions import BERTSCORE_LAYERS
from tgcl.models.revisions import BERTSCORE_MODEL
from tgcl.models.revisions import pinned_snapshot
from tgcl.models.revisions import revision

# Repository root (tgcl/evaluation/<this file>): relative image paths resolve
# against it.
REPO_ROOT = Path(__file__).resolve().parents[2]

MAX_WORDS = 10  # paper: at most 10 grounded words per concept
CLIP_WEIGHT = 2.5  # CLIPScore w (Hessel et al.)
ACTIVATION_RULES = {
    "snmf": "lasso",
    "pca": "projection",
    "simple": "kmeans",
}


# ------------------------------------------------------------------ concepts
# and words


def paper_valid_word(word: str, corpus: set, stopwords: set) -> bool:
    """The original metrics.utils.valid_word: English (NLTK words), > 2 letters,
    not a stop word -- or a number. (This repo's copy relaxed the length to >
    0.)"""
    word = word.lower().strip()
    return (
        word in corpus and len(word) > 2 and word not in stopwords
    ) or is_number(word)


def concept_words(words: Sequence[str], valid) -> List[str]:
    """Grounded words as the paper uses them: valid words, lower case, unique,
    at most 10."""
    out = []
    for w in words:
        w = str(w).lower().strip()
        if valid(w) and w not in out:
            out.append(w)
    return out[:MAX_WORDS]


def caption_for(words: Sequence[str]) -> str:
    """The original img_clipscore candidate: words joined by ', ' and a final
    '.'."""
    return ", ".join(words) + "."


def activations(
    x: np.ndarray, concepts: np.ndarray, method: str, alpha: float
) -> np.ndarray:
    """(N, K) concept activations of representations x (N, D) on the bank
    concepts (K, D)."""
    rule = ACTIVATION_RULES.get(method)
    if rule == "lasso":
        return sparse_encode(
            x,
            concepts,
            algorithm="lasso_cd",
            alpha=alpha,
            positive=True,
            max_iter=1000,
        )
    if rule == "projection":
        return x @ concepts.T
    if rule == "kmeans":
        return 1.0 / (
            1.0 + np.linalg.norm(x[:, None, :] - concepts[None, :, :], axis=-1)
        )
    raise ValueError(
        f"no activation rule for decomposition '{method}' (needs its "
        "fitted model)"
    )


def random_words(
    lm_head_weight: torch.Tensor,
    decode,
    desired: Sequence[int],
    valid,
    seed: int,
) -> List[List[str]]:
    """Per concept: as many distinct words as the concept has grounded words,
    decoded from a random direction (torch.rand, as the original
    get_random_words) and filtered like grounded words.

    The tokens are taken in the direction's ranking until n words pass the
    filter. The original cut the ranking at 10 n tokens, so with a large
    vocabulary (Qwen: 152k tokens, many fragments and non-Latin tokens) a
    concept could get no random word at all and its random baseline was NaN.
    """
    gen = torch.Generator().manual_seed(seed)
    out = []
    for n in desired:
        direction = torch.rand(1, lm_head_weight.shape[1], generator=gen)
        words: List[str] = []
        if n > 0:
            for t in (
                (direction @ lm_head_weight.T)[0]
                .argsort(descending=True)
                .tolist()
            ):
                w = decode(t).lower().strip()
                if valid(w) and w not in words:
                    words.append(w)
                    if len(words) == n:
                        break
        out.append(words)
    return out


# ------------------------------------------------------------------
# explanations


def first_token_rows(
    results: List[Dict[str, Any]], ignore_words: frozenset
) -> List[Tuple[int, str, int]]:
    """(result index, image path, activation index) of each result's first
    scored answer token."""
    rows = []
    for i, r in enumerate(results):
        for tok in r.get("per_token_concepts") or []:
            word = re.sub(
                r"[^a-z0-9]", "", str(tok.get("token_text", "")).lower()
            )
            if (
                word
                and word not in ignore_words
                and tok.get("activation_index") is not None
            ):
                rows.append(
                    (i, r.get("image_path", ""), int(tok["activation_index"]))
                )
                break
    return rows


# ------------------------------------------------------------------ CLIP


class Clip:
    """CLIP ViT-B/32 with the original CLIPScore image transform and caption
    prefix."""

    def __init__(self, device: str):
        import clip

        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        self.device = device
        self.dtype = next(self.model.parameters()).dtype

    def images(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        """Unit CLIP embeddings of images (paths or refs), one row each."""
        loader = torch.utils.data.DataLoader(
            CLIPImageDataset(paths), batch_size=batch_size, shuffle=False
        )
        with torch.no_grad():
            feats = [
                self.model.encode_image(b["image"].to(self.device, self.dtype))
                .float()
                .cpu()
                .numpy()
                for b in loader
            ]
        return _unit(np.vstack(feats))

    def captions(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Unit CLIP embeddings of texts, one row each."""
        loader = torch.utils.data.DataLoader(
            CLIPCaptionDataset(texts), batch_size=batch_size, shuffle=False
        )
        with torch.no_grad():
            feats = [
                self.model.encode_text(b["caption"].to(self.device))
                .float()
                .cpu()
                .numpy()
                for b in loader
            ]
        return _unit(np.vstack(feats))


def _unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm."""
    return x / np.sqrt(np.sum(x**2, axis=-1, keepdims=True))


def clip_scores(
    image_feats: np.ndarray, caption_feats: np.ndarray, top: np.ndarray
) -> np.ndarray:
    """(N, k) CLIPScore of each image vs the captions of its top-k concepts
    (concept rank order)."""
    return CLIP_WEIGHT * np.clip(
        np.einsum("nd,nkd->nk", image_feats, caption_feats[top]), 0, None
    )


# ------------------------------------------------------------------ BERTScore


def merged_words(word_sets: List[List[str]]) -> List[List[str]]:
    """Per concept, its words merged into one candidate text ("w1, w2, ...");
    none if no words."""
    return [[caption_for(w)] if w else [] for w in word_sets]


def bert_concept_scores(
    phrases: List[List[str]], captions: List[List[str]], top: np.ndarray, scorer
) -> np.ndarray:
    """(N, k): per sample and top concept, max BERTScore F1 over (phrase,
    caption) pairs. Unique pairs are scored once in one batched call."""
    pairs = sorted(
        {
            (p, c)
            for n in range(len(top))
            for ci in top[n]
            for p in phrases[ci]
            for c in captions[n]
        }
    )
    f1 = {}
    if pairs:
        _, _, values = scorer.score(
            [p for p, _ in pairs],
            [c for _, c in pairs],
            verbose=False,
            batch_size=256,
        )
        f1 = dict(zip(pairs, (float(v) for v in values)))
    out = np.full(top.shape, np.nan)
    for n in range(len(top)):
        for j, ci in enumerate(top[n]):
            vals = [f1[(p, c)] for p in phrases[ci] for c in captions[n]]
            if vals:
                out[n, j] = max(vals)
    return out


# ------------------------------------------------------------------ main


def topk_summary(
    scores: np.ndarray, ks: Sequence[int]
) -> Dict[int, Tuple[float, float, int]]:
    """k -> (mean, std, n) of the per-sample mean over the top-k concepts (NaN
    samples skipped)."""
    out = {}
    for k in ks:
        with (
            warnings.catch_warnings()
        ):  # rows without any score (no captions) give NaN by design
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = (
                np.nanmean(scores[:, :k], axis=1)
                if scores.size
                else np.array([])
            )
        vals = vals[~np.isnan(vals)]
        out[k] = (
            float(vals.mean()) if len(vals) else float("nan"),
            float(vals.std()) if len(vals) else float("nan"),
            int(len(vals)),
        )
    return out


def load_lm_head(model_name: str) -> torch.Tensor:
    """Output projection weight (V, D) from the model's safetensors (tied
    embeddings if no lm_head)."""
    from safetensors import safe_open

    snapshot = Path(pinned_snapshot(model_name))
    with open(snapshot / "model.safetensors.index.json", encoding="utf-8") as f:
        index = json.load(f)["weight_map"]
    keys = (
        [k for k in index if k.endswith("lm_head.weight")]
        or [
            k for k in index if k.endswith("language_model.embed_tokens.weight")
        ]
        or [k for k in index if k.endswith("embed_tokens.weight")]
    )  # tied embeddings (e.g. Gemma)
    if not keys:
        raise KeyError(f"no lm_head/embed_tokens weight in {model_name}")
    key = keys[0]
    with safe_open(str(snapshot / index[key]), "pt") as f:
        return f.get_tensor(key).float()


def main() -> None:
    apply_determinism_setting()  # DETERMINISTIC=1
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--json_path",
        required=True,
        help="vlm_explanations.json (with raw activations)",
    )
    ap.add_argument(
        "--concept_path", required=True, help="combined concept bank (.pth)"
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--model_name",
        default=os.environ.get("VLM_MODEL", "google/gemma-3n-E4B-it"),
    )
    ap.add_argument("--max_k", type=int, default=3)
    ap.add_argument(
        "--dl_alpha",
        type=float,
        default=float(os.environ.get("DL_ALPHA", 1.0)),
        help="lasso alpha of the snmf transform (the decomposition's alpha)",
    )
    ap.add_argument(
        "--ignore_words",
        default="",
        help="answer words that are not the token of interest (e.g. no,yes)",
    )
    ap.add_argument(
        "--captions_json",
        default=None,
        help="{image file name: [captions]} for BERTScore",
    )
    ap.add_argument(
        "--phrases_json",
        default=None,
        help="{caption of words: [phrases]} from "
        "tgcl/evaluation/word_alignment_phrases.py",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        default=os.environ.get(
            "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.load(open(args.json_path, encoding="utf-8"))
    results = payload["results"]
    if payload.get("activations_kind") != "hidden_state":
        raise ValueError(
            f"{args.json_path}: no raw activations; re-run pipeline step 6"
        )
    # The states next to the JSON win over its stored (possibly foreign) path.
    sibling = os.path.splitext(args.json_path)[0] + "_activations.npy"
    acts = np.load(
        sibling if os.path.exists(sibling) else payload.get("activations_path")
    )
    bank = torch.load(args.concept_path, map_location="cpu", weights_only=False)
    concepts = torch.as_tensor(bank["concepts"]).float().numpy()
    method = str(bank.get("decomposition_method", "snmf"))
    if method not in ACTIVATION_RULES:
        print(
            f"Skip concept-word alignment: decomposition '{method}' has "
            "no activation rule without its fitted model"
        )
        return
    stop = set(get_stopwords(GIST_FILE_PATH))
    corpus = set(english_words())

    def valid(w):
        return paper_valid_word(w, corpus, stop)

    grounded = [concept_words(ws, valid) for ws in bank["text_grounding"]]

    ignore = frozenset(
        w.strip().lower() for w in args.ignore_words.split(",") if w.strip()
    )
    rows = first_token_rows(results, ignore)
    ks = tuple(range(1, min(args.max_k, len(concepts)) + 1))
    act = activations(
        acts[[a for _, _, a in rows]], concepts, method, args.dl_alpha
    )
    top = np.argsort(-act, axis=1)[:, : ks[-1]]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=revision(args.model_name)
    )
    rnd = random_words(
        load_lm_head(args.model_name),
        lambda t: tokenizer.decode([t], skip_special_tokens=True),
        [max(1, len(w)) for w in grounded],
        valid,
        args.seed,
    )
    (out_dir / "word_alignment_words.json").write_text(
        json.dumps({"grounded": grounded, "random": rnd}, indent=1)
    )

    clip = Clip(args.device)
    img = clip.images(
        [
            p
            if is_memory_ref(p) or Path(p).is_absolute()
            else str(REPO_ROOT / p)
            for _, p, _ in rows
        ]
    )
    scores = {
        "word_clipscore": clip_scores(
            img, clip.captions([caption_for(w) for w in grounded]), top
        ),
        "random_word_clipscore": clip_scores(
            img, clip.captions([caption_for(w) for w in rnd]), top
        ),
    }

    if args.captions_json:
        from bert_score import BERTScorer

        caps_by_file = json.load(open(args.captions_json, encoding="utf-8"))
        # a crop keeps the captions of its image; a grid has none
        caps = [
            caps_by_file.get(Path(source_image(p) or "").name, [])
            for _, p, _ in rows
        ]
        scorer = BERTScorer(
            model_type=pinned_snapshot(BERTSCORE_MODEL),
            num_layers=BERTSCORE_LAYERS,
            lang="en",
            device=args.device,
        )
        # The concept's words merged into one text (the CLIPScore caption) vs
        # the captions.
        scores["word_bertscore"] = bert_concept_scores(
            merged_words(grounded), caps, top, scorer
        )
        scores["random_word_bertscore"] = bert_concept_scores(
            merged_words(rnd), caps, top, scorer
        )
    if args.captions_json and args.phrases_json:
        phrases = json.load(open(args.phrases_json, encoding="utf-8"))
        missing = [
            ", ".join(w) for w in grounded + rnd if ", ".join(w) not in phrases
        ]
        if missing:
            raise KeyError(
                f"{len(missing)} word sets have no phrases (run "
                f"eval/word_alignment_phrases.py), e.g. {missing[0]}"
            )
        scores["phrase_bertscore"] = bert_concept_scores(
            [phrases[", ".join(w)] for w in grounded], caps, top, scorer
        )
        scores["random_phrase_bertscore"] = bert_concept_scores(
            [phrases[", ".join(w)] for w in rnd], caps, top, scorer
        )

    with open(
        out_dir / "word_alignment_table.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["rank_k", "metric", "mean", "std", "n_samples", "activation_rule"]
        )
        for name, s in scores.items():
            for k, (m, sd, n) in topk_summary(s, ks).items():
                w.writerow([k, name, m, sd, n, ACTIVATION_RULES.get(method)])
    with open(
        out_dir / "word_alignment_per_sample.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["sample", "image_path", "top_concepts"]
            + [f"{n}_top{j + 1}" for n in scores for j in range(len(ks))]
        )
        for r, (i, path, _) in enumerate(rows):
            w.writerow(
                [i, path, " ".join(str(c) for c in top[r])]
                + [
                    ("" if np.isnan(v) else float(v))
                    for per_name in scores.values()
                    for v in per_name[r]
                ]
            )
    for name, s in scores.items():
        print(name, {k: round(v[0], 4) for k, v in topk_summary(s, ks).items()})


if __name__ == "__main__":
    main()
