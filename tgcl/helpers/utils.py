"""Forward hooks that record a VLM's hidden states, and what is saved from them.

Step 4 (tgcl.save_features) registers hooks on the chosen module; during
generation they collect every module output in HIDDEN_STATES. After each batch,
get_hidden_states reduces those outputs to one vector per sample (the class
word in the answer, a fixed token, or a mean) and save_hidden_states_to_file
writes them with the sample metadata.
"""

import argparse
import contextlib
from functools import partial
import logging
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch

__all__ = [
    "register_hooks",
    "clear_forward_hooks",
    "clear_hooks_variables",
    "hooks_postprocessing",
    "set_seed",
    "setup_hooks",
    "resolve_layer_path",
]


# Dictionary to store hidden states
HIDDEN_STATES = {}
# While True, save_hidden_states hooks record nothing
# (hidden_state_hooks_paused).
_HOOKS_PAUSED = False


@contextlib.contextmanager
def hidden_state_hooks_paused():
    """Run a forward / generate whose hidden states are not needed:
    save_hidden_states hooks skip
    their (GPU -> CPU) copy. Used by step 4's forced-answer mode around the
    deciding generation.
    """
    global _HOOKS_PAUSED
    previous, _HOOKS_PAUSED = _HOOKS_PAUSED, True
    try:
        yield
    finally:
        _HOOKS_PAUSED = previous


def get_batch_lengths(model_output):
    """
    Given model_output as a list of lists (batch dimension), return a list of
    lengths for each batch item. Handles both tensors and lists.
    """
    lengths = []
    for item in model_output:
        if torch.is_tensor(item):
            lengths.append(item.shape[0])
        else:
            lengths.append(len(item))
    return lengths


def mean_skip_special_tokens(
    v: torch.Tensor, pred_tokens: torch.Tensor, special_ids: set
) -> torch.Tensor:
    """
    Compute mean of hidden states per batch, skipping embeddings for tokens in
    special_ids.
    Args:
        v: (B, L, D) hidden states
        pred_tokens: (B, L) token ids
        special_ids: set of token ids to skip
    Returns:
        (B, 1, D) mean hidden states per batch

    Padding positions (token id -1) never count. If every real token is special,
    the mean over all real tokens is used instead of a zero vector.
    """
    mean_vecs = []
    special = torch.tensor(
        sorted(special_ids), dtype=pred_tokens.dtype, device=pred_tokens.device
    )
    for b in range(v.shape[0]):
        row_hidden = v[b]
        row_pred = pred_tokens[b]
        real = row_pred != -1
        keep_mask = real & ~torch.isin(row_pred, special)
        if not keep_mask.any():
            keep_mask = real
            if real.any():
                warnings.warn(
                    "All generated tokens are special tokens; averaging "
                    "over them."
                )
        kept = row_hidden[keep_mask.to(row_hidden.device)]
        if kept.shape[0] > 0:
            mean_vecs.append(kept.mean(dim=0, keepdim=True))
        else:  # no real token at all (empty answer)
            mean_vecs.append(
                torch.zeros((1, v.shape[2]), dtype=v.dtype, device=v.device)
            )
    return torch.cat(mean_vecs, dim=0).unsqueeze(1)


def set_seed(seed_value=42, deterministic=True, disable_tf32=False):
    """Seed Python, NumPy and PyTorch (all GPUs).

    Args:
        seed_value: The seed.
        deterministic: Deterministic cuDNN, no benchmarking.
        disable_tf32: Also turn TF32 matmuls off (the explainer and the
            deletion curves do when deterministic; feature saving keeps TF32).
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    apply_determinism_setting()


def apply_determinism_setting() -> bool:
    """Strict GPU determinism when DETERMINISTIC=1; otherwise nothing.

    Turns on torch's deterministic algorithms (warn_only: an op without one
    warns instead of failing the run), cuBLAS's deterministic workspace,
    deterministic cuDNN and no TF32. A run is then bitwise reproducible on the
    same GPU model, driver and library versions, at some cost in speed. Call it
    before the first CUDA operation (set_seed does).

    Returns:
        Whether the setting was applied.
    """
    if os.environ.get("DETERMINISTIC", "0") != "1":
        return False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return True


def update_dict_of_list(
    item: Dict[str, Any], data: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge a batched item dict into an accumulator of lists.
    - If item[k] is a list/tuple (batched), extend the accumulator list.
    - Otherwise, append the single value.
    Also ensures that any torch.Tensors are detached and moved to CPU before
    storing.
    """

    def _to_cpu(x: Any) -> Any:
        return x.detach().cpu() if torch.is_tensor(x) else x

    for k, v in item.items():
        # Case 1: batched sequences
        if isinstance(v, (list, tuple)):
            values = [_to_cpu(x) for x in v]
            if k in data:
                data[k].extend(values)
            else:
                data[k] = list(values)

        # Case 2: nested mapping (e.g., dict of lists/sequences)
        elif isinstance(v, dict):
            for ek, ev in v.items():
                if k not in data:
                    data[k] = {}
                if isinstance(ev, (list, tuple)):
                    proc = [_to_cpu(x) for x in ev]
                else:
                    # Fallback: try to iterate (e.g., 1D tensor) else treat as
                    # single
                    if torch.is_tensor(ev):
                        try:
                            proc = [_to_cpu(x) for x in list(ev)]
                        except TypeError:
                            proc = [_to_cpu(ev)]
                    else:
                        try:
                            proc = [_to_cpu(x) for x in ev]  # type: ignore
                        except TypeError:  # not iterable: one value
                            proc = [_to_cpu(ev)]

                if ek in data[k]:
                    data[k][ek].extend(proc)
                else:
                    data[k][ek] = list(proc)

        # Case 3: single value
        else:
            val = _to_cpu(v)
            if k in data:
                data[k].append(val)
            else:
                data[k] = [val]
    return data


def fmatch(name: str, patterns: List[str], exact_match: bool = False) -> bool:
    if exact_match:
        return name in patterns
    else:
        # Convert patterns with '*' to proper regex expressions (where * means
        # "any sequence of characters")
        regex_patterns = [
            re.compile(re.sub(r"\*", ".*", pattern)) for pattern in patterns
        ]
        return any([regex.search(name) for regex in regex_patterns])


def compute_time_left(start_time, iteration: int, num_iterations: int):
    elapsed_time = time.time() - start_time  # Time spent so far
    avg_time_per_iter = elapsed_time / iteration  # Average time per iteration
    remaining_iters = num_iterations - iteration
    time_left = avg_time_per_iter * remaining_iters  # Estimated time left
    return time_left / 60


def save_hidden_states(module_name: str = "", **kwargs: Any):
    """
    Save module output hidden states. In case of autoregressive, make sure the
    kv caching is enabled.
    """
    del kwargs  # Unused: every hook factory gets the same options.

    def hook(module, inputs, output):
        """Append the module's output (detached, on CPU) to HIDDEN_STATES."""
        del inputs  # Unused: the forward-hook signature.
        if _HOOKS_PAUSED:
            return
        if isinstance(output, tuple):  # e.g residual streams output is a tuple
            output = output[0]
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            # AltUp-style decoder layers (e.g. Gemma3n's
            # Gemma3nTextDecoderLayer) return a stacked tensor of shape
            # (altup_num_inputs, batch, seq, hidden) instead of the
            # conventional (batch, seq, hidden) hidden state. Select the
            # "active" prediction stream (the module's own config carries
            # altup_active_idx) so downstream code -- which assumes 3D
            # hidden states -- doesn't silently treat the batch/seq axes
            # incorrectly. This matters whenever LOGIT_LENS_LAYER_SELECTION
            # picks a raw decoder layer (not the final norm) for extraction.
            altup_idx = getattr(
                getattr(module, "config", None), "altup_active_idx", 0
            )
            output = output[altup_idx]
        output = output.detach().cpu()
        if module_name in HIDDEN_STATES:
            HIDDEN_STATES[module_name].append(output)
        else:
            HIDDEN_STATES[module_name] = [output]

    return hook


TokenSequences = List[Tuple[int, ...]]


def encode_concept_token_sequences(word: str, tokenizer: Any) -> TokenSequences:
    """All token-id sequences that spell the concept `word` in a generated
    answer.

    Covers the word as given, lowercased and capitalized, each singular and with
    a plural "s", each encoded both at the start of a text and after a space
    (tokenizers usually give mid-sentence words a different id). Underscores are
    read as spaces, so the tag "gas_station" matches the answer "gas station".
    Leading ids that decode to pure whitespace (a stray space piece some
    tokenizers emit for " word") are dropped, so every sequence starts at the
    word itself.

    Multi-token words yield multi-id sequences; the token-of-interest hook
    averages the hidden states over the whole matched sequence.
    """
    base = word.replace("_", " ").strip()
    if not base:
        raise ValueError(f"Empty concept word: {word!r}")
    forms = {base, base.lower(), base.capitalize()}
    forms |= {form + "s" for form in forms}

    sequences = set()
    for form in forms:
        for text in (form, " " + form):
            ids = list(tokenizer.encode(text, add_special_tokens=False))
            while ids and not tokenizer.decode([ids[0]]).strip():
                ids.pop(0)
            if ids:
                sequences.add(tuple(int(i) for i in ids))
    return sorted(sequences)


def find_concept_span(
    pred_tokens: Union[torch.Tensor, List[int]], sequences: TokenSequences
) -> Optional[Tuple[int, int]]:
    """Half-open [start, end) span of the first occurrence of any sequence.

    The earliest start wins; at equal starts the longest sequence wins, so a
    plural ("apple" + "s") is taken over its singular prefix. Returns None if no
    sequence occurs.
    """
    row = (
        pred_tokens.tolist()
        if torch.is_tensor(pred_tokens)
        else list(pred_tokens)
    )
    best: Optional[Tuple[int, int]] = None
    # pylint: disable=unsubscriptable-object  (best is a tuple once set)
    for seq in sequences:
        n = len(seq)
        if n == 0:
            continue
        first = seq[0]
        for start in range(len(row) - n + 1):
            if best is not None and start > best[0]:
                break
            if row[start] == first and tuple(row[start : start + n]) == seq:
                if best is None or start < best[0] or n > best[1] - best[0]:
                    best = (start, start + n)
                break
    return best


def _as_token_sequences(
    token_sequences: Optional[TokenSequences],
    token_of_interest_idx: Union[int, List[int], torch.Tensor, None],
) -> TokenSequences:
    """Normalise the two accepted inputs to a list of id sequences.

    `token_of_interest_idx` is the legacy form (single ids, e.g. from the
    --token_of_interest_idx CLI flag); each id becomes a one-token sequence.
    """
    if token_sequences is not None:
        return [tuple(int(i) for i in seq) for seq in token_sequences]
    if token_of_interest_idx is None:
        raise ValueError("Provide token_sequences or token_of_interest_idx.")
    if torch.is_tensor(token_of_interest_idx):
        ids = token_of_interest_idx.flatten().tolist()
    elif isinstance(token_of_interest_idx, int):
        ids = [token_of_interest_idx]
    else:
        ids = list(token_of_interest_idx)
    return [(int(i),) for i in ids]


def extract_token_of_interest_states(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    token_sequences: Optional[TokenSequences] = None,
    token_sequences_per_sample: Optional[List[TokenSequences]] = None,
    token_of_interest_start_token: Union[
        int, List[int], Tuple[int, ...], torch.Tensor
    ] = 0,
    token_of_interest_idx: Union[int, List[int], torch.Tensor, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hidden state of the concept word in each sample's answer.

    For every row, finds the first occurrence of the concept (any sequence in
    `token_sequences`, or the row's own entry of `token_sequences_per_sample`)
    in `pred_tokens` and averages `tokens` over the matched positions. A
    single-token word therefore yields that one position; a multi-token word
    yields the mean over all of its tokens. Surrounding answer tokens such as
    "No"/"Yes" are never part of the span.

    Returns:
        (B, 1, D) states, and a (B,) bool mask that is False where the concept
        was not found (those rows hold the state at position 0 and must be
        dropped downstream).
    """

    # Optionally crop sequences starting from a given token position. If a batch
    # of
    # start indices is provided, slice per-sample and pad back to a uniform
    # length.
    if isinstance(token_of_interest_start_token, (list, tuple, torch.Tensor)):
        # Align lengths first so per-sample slicing uses the same base lengths
        if pred_tokens.shape[1] > tokens.shape[1]:
            pred_tokens = pred_tokens[:, -tokens.shape[1] :]
        elif pred_tokens.shape[1] < tokens.shape[1]:
            tokens = tokens[:, -pred_tokens.shape[1] :]

        B, L, D = tokens.shape
        # Convert to tensor of shape (B,)
        starts = (
            token_of_interest_start_token
            if isinstance(token_of_interest_start_token, torch.Tensor)
            else torch.tensor(token_of_interest_start_token)
        )
        assert starts.numel() == B, (
            f"token_of_interest_start_token must have length {B}, got "
            f"{starts.numel()}"
        )
        starts = starts.to(torch.long)

        # Slice per-sample, allowing negative indexing like python: -k means L-k
        tokens_list: List[torch.Tensor] = []
        preds_list: List[torch.Tensor] = []
        for b in range(B):
            s = starts[b].item()
            s = s if s >= 0 else max(0, L + s)
            s = min(max(0, s), L)  # clamp to [0, L]
            tokens_list.append(tokens[b, s:, :])
            preds_list.append(pred_tokens[b, s:])

        # Pad back to uniform length
        max_len = max(t.shape[0] for t in tokens_list) if tokens_list else 0
        if max_len == 0:
            # Degenerate case; keep shapes consistent
            tokens = tokens[:, :0, :]
            pred_tokens = pred_tokens[:, :0]
        else:
            padded_tokens = tokens.new_zeros((B, max_len, D))
            padded_preds = pred_tokens.new_full((B, max_len), fill_value=-1)
            for b in range(B):
                l = tokens_list[b].shape[0]
                if l > 0:
                    padded_tokens[b, :l, :] = tokens_list[b]
                    padded_preds[b, :l] = preds_list[b]
            tokens, pred_tokens = padded_tokens, padded_preds
    elif token_of_interest_start_token != 0:
        # Scalar start index: simple slice for all samples
        tokens = tokens[:, token_of_interest_start_token:]
        pred_tokens = pred_tokens[:, token_of_interest_start_token:]

    # Concider only text, no preds tokens for image tokens
    if pred_tokens.shape[1] > tokens.shape[1]:
        pred_tokens = pred_tokens[
            :,
            -tokens.shape[1] :,
            # e.g. in case of language_model.lm_head only the hidden states for
            # generated tokens are saved
        ]
    elif pred_tokens.shape[1] < tokens.shape[1]:
        tokens = tokens[:, -pred_tokens.shape[1] :]

    batch_size = pred_tokens.shape[0]
    if token_sequences_per_sample is not None:
        # P_bin_shuf: each sample was asked about a different (shuffled)
        # concept, so each row is searched for the word IT was asked about,
        # not the batch's tag word (which the shuffled prompt never mentions).
        if len(token_sequences_per_sample) != batch_size:
            raise ValueError(
                "token_sequences_per_sample needs one entry per batch "
                "row, got "
                f"{len(token_sequences_per_sample)} for batch size {batch_size}"
            )
        row_sequences = [
            _as_token_sequences(seqs, None)
            for seqs in token_sequences_per_sample
        ]
    else:
        shared = _as_token_sequences(token_sequences, token_of_interest_idx)
        row_sequences = [shared] * batch_size

    found = torch.zeros(batch_size, dtype=torch.bool)
    if tokens.shape[1] == 0:
        empty = tokens.new_zeros((batch_size, 1, tokens.shape[2]))
        return empty, found.to(pred_tokens.device)

    selected = []
    for b in range(batch_size):
        span = find_concept_span(pred_tokens[b], row_sequences[b])
        if span is None:
            selected.append(tokens[b, 0])
        else:
            selected.append(tokens[b, span[0] : span[1]].mean(dim=0))
            found[b] = True
    v_selected = torch.stack(selected, dim=0).unsqueeze(1)
    return v_selected, found.to(pred_tokens.device)


def _token_matrix(tokens) -> torch.Tensor:
    """Token ids as a (batch, length) long tensor; a list of per-sample
    sequences is right-padded with -1."""
    if isinstance(tokens, list):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(t, dtype=torch.long) for t in tokens],
            batch_first=True,
            padding_value=-1,
        )
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    return tokens.to(dtype=torch.long)


def _mean_over_answer(
    states: torch.Tensor, tokens, tokenizer: Any, extra_special_ids=()
) -> torch.Tensor:
    """Mean of the states over the non-special tokens.

    Args:
        states: Hidden states, (batch, length, dim).
        tokens: The token ids those states belong to (list or tensor); when
            longer than the states, only their last positions are used.
        tokenizer: Supplies the special token ids to skip.
        extra_special_ids: Further ids to skip (e.g. 106, a Qwen end token).

    Returns:
        (batch, 1, dim) mean states.
    """
    special_ids = set(tokenizer.all_special_ids) | set(extra_special_ids)
    token_ids = _token_matrix(tokens).to(states.device)
    if token_ids.shape[1] > states.shape[1]:
        token_ids = token_ids[:, -states.shape[1] :]
    elif token_ids.shape[1] < states.shape[1]:
        states = states[:, -token_ids.shape[1] :, :]
    return mean_skip_special_tokens(states, token_ids, special_ids)


def get_hidden_states(
    token_idx: Optional[int] = None,
    extract_token_of_interest: bool = False,
    token_of_interest_start_token: int = 0,
    save_only_generated_tokens: bool = False,
    mean: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reduce the recorded hidden states of a batch to one vector per sample.

    Exactly one reduction applies, in this order: a fixed position
    (token_idx), the concept word in the answer (extract_token_of_interest), or
    all positions (optionally only the generated ones, optionally averaged).

    Args:
        token_idx: Keep the state at this sequence position.
        extract_token_of_interest: Mean over the tokens of the concept word in
            the answer (token_sequences / token_of_interest_idx in kwargs).
        token_of_interest_start_token: Where the search for the word starts.
        save_only_generated_tokens: Use only the generated positions.
        mean: Average over the kept positions, skipping special tokens.
        **kwargs: Batch context from the explainer/feature saver: model_output,
            model_generated_output, tokenizer, image, token_sequences, ...

    Returns:
        {"hidden_states": {module name: [one tensor per sample]}}, plus
        "token_of_interest_mask" and "image" when the concept word is searched.
    """
    hidden_states = {}
    output = {}
    for k, v in HIDDEN_STATES.items():
        if isinstance(v, list) and len(v) > 1:
            v = torch.cat(v, dim=1)
            # v = v / v.norm(p=2, dim=1, keepdim=True)
            # Extra added for balancing distance avoiding neighbour tokens
        else:
            v = v[0]
        if token_idx is not None:
            # clone: a slice keeps the whole sequence's storage alive, and
            # torch.save writes all of it (~270x the one token's size).
            v = v[:, token_idx, :].unsqueeze(1).clone()
        elif extract_token_of_interest:

            if save_only_generated_tokens:
                token_of_interest_start_token_batch = []

                batch_lengths = get_batch_lengths(
                    kwargs["model_generated_output"]
                )
                for length in batch_lengths:
                    start_idx_generated_tokens = -length
                    token_of_interest_start_token_batch.append(
                        start_idx_generated_tokens
                    )
                toi_start = token_of_interest_start_token_batch
            else:
                # Use the provided scalar or batch value from function args
                toi_start = token_of_interest_start_token

            pred_tokens_tensor = _token_matrix(kwargs["model_output"])

            # P_bin_shuf: each sample's prompt asked about a different
            # (shuffled) concept -- kwargs["prompt_concept"] carries that
            # per-sample word (see JSONDataset.create_dataset()). Build a
            # per-sample token id set so each row is searched for the word
            # IT was actually asked about, not the batch's tag-level word
            # (which the shuffled prompt never mentions).
            prompt_concepts = kwargs.get("prompt_concept", None)
            sequences_per_sample = None
            tag_word = kwargs.get("token_of_interest", None)
            hook_tokenizer = kwargs.get("tokenizer", None)
            if (
                prompt_concepts
                and hook_tokenizer is not None
                and any(pc != tag_word for pc in prompt_concepts)
            ):
                encoded_cache: Dict[str, TokenSequences] = {}
                sequences_per_sample = []
                for pc in prompt_concepts:
                    if pc not in encoded_cache:
                        encoded_cache[pc] = encode_concept_token_sequences(
                            pc, hook_tokenizer
                        )
                    sequences_per_sample.append(encoded_cache[pc])

            v, token_of_interest_mask = extract_token_of_interest_states(
                tokens=v,
                pred_tokens=pred_tokens_tensor,
                token_sequences=kwargs.get("token_sequences", None),
                token_sequences_per_sample=sequences_per_sample,
                token_of_interest_start_token=toi_start,
                token_of_interest_idx=kwargs.get("token_of_interest_idx", None),
            )
            output["token_of_interest_mask"] = token_of_interest_mask
            output["image"] = kwargs["image"]
        else:
            if save_only_generated_tokens:
                # Slice v to only generated tokens using per-sample starts
                mgo = kwargs.get("model_generated_output", None)

                B, L, D = v.shape
                if isinstance(mgo, list):
                    batch_lengths = get_batch_lengths(mgo)
                    starts = torch.tensor(
                        [-int(length) for length in batch_lengths],
                        dtype=torch.long,
                    )
                elif torch.is_tensor(mgo):
                    starts = torch.full((B,), -int(mgo.shape[1]))
                else:
                    starts = torch.zeros((B,), dtype=torch.long)

                tokens_list: List[torch.Tensor] = []
                for b in range(B):
                    s = starts[b].item()
                    s = s if s >= 0 else max(0, L + s)
                    s = min(max(0, s), L)
                    tokens_list.append(v[b, s:, :])

                max_len = (
                    max(t.shape[0] for t in tokens_list) if tokens_list else 0
                )
                if max_len == 0:
                    v = v[:, :0, :]
                else:
                    padded_tokens = v.new_zeros((B, max_len, D))
                    for b in range(B):
                        l = tokens_list[b].shape[0]
                        if l > 0:
                            padded_tokens[b, :l, :] = tokens_list[b]
                    v = padded_tokens

                if mean:
                    v = _mean_over_answer(
                        v,
                        kwargs.get("model_generated_output"),
                        kwargs["tokenizer"],
                        extra_special_ids=(106,),
                    )
            else:
                if mean:
                    v = _mean_over_answer(
                        v, kwargs.get("model_output"), kwargs["tokenizer"]
                    )

        hidden_states[k] = [v[i] for i in range(v.shape[0])]
    output["hidden_states"] = hidden_states
    return output


def save_hidden_states_to_file(
    data: Dict[str, Any],
    data_keys: Optional[List[str]] = None,
    hook_name: str = "",
    args: Optional[argparse.Namespace] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Write the selected entries of a batch's results to the features folder.

    The file is <save_dir>/features/<hook_name>_<save_filename>.pth, written
    to a temporary name first so an interrupted run leaves no partial file.

    Args:
        data: Results of the hook return function and the sample metadata.
        data_keys: Entries to save (default: hidden_states).
        hook_name: Hook whose results these are (part of the file name).
        args: Pipeline arguments (save_dir, save_filename).
        logger: Optional logger.
    """
    if data_keys is None:
        data_keys = ["hidden_states"]

    saved_data = {}

    for data_key in data.keys():
        if data_key in data_keys:
            assert (
                data_key in data
            ), f"{data_key} not found in data, there is only: {data.keys()}"

            saved_data[data_key] = data[data_key]  # List[Any]
    file_name = os.path.join(
        args.save_dir, "features", f"{hook_name}_{args.save_filename}.pth"
    )
    # Ensure parent directory exists before saving
    parent_dir = os.path.dirname(file_name)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        if logger is not None:
            logger.info(
                f"Created directory {parent_dir} for saving hidden states."
            )

    # Write then rename: a run killed mid-save leaves no truncated .pth that a
    # resumed run would take for a finished tag.
    tmp_name = f"{file_name}.tmp"
    torch.save(saved_data, tmp_name)
    os.replace(tmp_name, file_name)
    if logger is not None:
        logger.info(f"Saving data to: {file_name}")


def feature_file_paths(args: argparse.Namespace) -> List[str]:
    """Files save_hidden_states_to_file writes for the current
    args.save_filename."""
    return [
        os.path.join(
            args.save_dir, "features", f"{hook_name}_{args.save_filename}.pth"
        )
        for hook_name in args.hook_names
        if "save_hidden_states" in hook_name
    ]


def save_analysis_to_file(
    data: Dict[str, Any],
    analysis_saving_path: str,
    data_keys: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save the given entries of an analysis result to <saving path>.pth.

    Args:
        data: Analysis results (e.g. text groundings of a concept bank).
        analysis_saving_path: Output path without the .pth suffix.
        data_keys: Entries to save (default: text_grounding); each must exist.
        logger: Optional logger.
    """
    if data_keys is None:
        data_keys = ["text_grounding"]

    saved_data = {}

    for data_key in data_keys:
        assert (
            data_key in data
        ), f"{data_key} not found in data, there is only: {data.keys()}"

        saved_data[data_key] = data[data_key]  # List[Any]
    file_name = f"{analysis_saving_path}.pth"
    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving analysis data to: {file_name}")


def register_hooks(
    model: torch.nn.Module,
    modules_to_hook: List[str],
    hook_name: str = "save_hidden_states",
    tokenizer: Any = None,
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
) -> Callable:
    """Register a hidden-state hook on every module matching modules_to_hook.

    Args:
        model: The VLM (a torch module).
        modules_to_hook: Module name patterns (see fmatch).
        hook_name: save_hidden_states, save_hidden_states_mean,
            save_hidden_states_given_token_idx or
            save_hidden_states_for_token_of_interest.
        tokenizer: Needed to find the concept word and special tokens.
        logger: Optional logger.
        args: Pipeline arguments (token_idx, token_of_interest, ...).

    Returns:
        The function that turns the recorded states into per-sample vectors
        (get_hidden_states with the hook's settings), or None for an unknown
        hook name (a warning is issued).
    """
    hook_function, hook_return_function = None, None

    if "save_hidden_states" == hook_name:
        # Save the hidden states of all tokens in the sequence
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            save_only_generated_tokens=args.save_only_generated_tokens,
            tokenizer=tokenizer,
        )
    elif "save_hidden_states_mean" == hook_name:
        # Save the hidden states of all tokens in the sequence
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            save_only_generated_tokens=args.save_only_generated_tokens,
            tokenizer=tokenizer,
            mean=True,
        )
    elif "save_hidden_states_given_token_idx" == hook_name:
        # Save the hidden states at given token index
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states, token_idx=args.token_idx
        )
    elif "save_hidden_states_for_token_of_interest" == hook_name:
        # Hidden state of the concept word in the answer: mean over all of its
        # tokens (case, plural and leading-space variants; see
        # encode_concept_token_sequences). An explicit --token_of_interest_idx
        # overrides the word with that single token id.
        token_of_interest = args.token_of_interest
        token_sequences = None
        if args.token_of_interest_idx is None:
            token_sequences = encode_concept_token_sequences(
                token_of_interest, tokenizer
            )

        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_sequences=token_sequences,
            token_of_interest_idx=args.token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            save_only_generated_tokens=args.save_only_generated_tokens,
            # Needed only for P_bin_shuf's per-sample token-of-interest
            # override (see get_hidden_states) -- harmless no-op otherwise.
            tokenizer=tokenizer,
            token_of_interest=token_of_interest,
        )
    else:
        warnings.warn(
            f"{hook_name} is not supported. No hooks attached to model."
        )
    if hook_function is not None:
        hooked_modules = []
        for name, module in model.named_modules():
            if fmatch(
                name,
                modules_to_hook,
                exact_match=args.exact_match_modules_to_hook,
            ):
                module.register_forward_hook(hook_function(module_name=name))
                hooked_modules.append(name)
        if logger is not None:
            logger.info(
                f"Apply {hook_name} to hooked_modules: {hooked_modules}"
            )

    return hook_return_function


def hooks_postprocessing(
    hook_name: str = "save_hidden_states",
    args: Optional[argparse.Namespace] = None,
) -> Callable:
    """The function that saves a hook's per-batch results (None if unknown).

    Args:
        hook_name: A save_hidden_states* hook name.
        args: Pipeline arguments (where to save).

    Returns:
        save_hidden_states_to_file with the entries to keep, or None.
    """
    hook_postprocessing_function = None
    if "save_hidden_states" in hook_name:
        data_keys = [
            "concept",
            "hidden_states",
            "image",
            "model_predictions",
            "bbox",
            "seg_mask_rle",
            "is_concept",
            "selected_layer",
            "logit_lens_selection",
        ]

        if "token_of_interest" in hook_name:
            data_keys.append("token_of_interest_mask")
        hook_postprocessing_function = partial(
            save_hidden_states_to_file,
            args=args,
            data_keys=data_keys,
            hook_name=hook_name,
        )
    else:
        warnings.warn(
            f"{hook_name} is not supported. No hooks attached to model."
        )

    return hook_postprocessing_function


def clear_forward_hooks(model: torch.nn.Module) -> None:
    for module in model.modules():
        # torch has no public call that removes every hook of a module.
        module._forward_hooks.clear()  # pylint: disable=protected-access


def clear_hooks_variables():
    global HIDDEN_STATES
    HIDDEN_STATES = {}


def setup_hooks(
    model: torch.nn.Module,
    modules_to_hook: List[str],
    hook_names: str,
    tokenizer: Any = None,
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
):
    """Register each named hook on its modules.

    Args:
        model: The VLM.
        modules_to_hook: One list of module patterns per hook name (None:
            nothing is hooked; only the saving functions are returned).
        hook_names: Hook names, in the order of modules_to_hook.
        tokenizer: Passed to the hooks.
        logger: Optional logger.
        args: Pipeline arguments.

    Returns:
        (return functions, saving functions), one of each per hook name.

    Raises:
        ValueError: Fewer module lists than hook names.
    """
    hook_return_functions, hook_postprocessing_functions = [], []
    if modules_to_hook is not None and len(modules_to_hook) < len(hook_names):
        raise ValueError(
            f"Got {len(hook_names)} hook_names but only {len(modules_to_hook)} "
            "modules_to_hook; every hook needs a module or no features "
            "are saved."
        )
    for i, hook_name in enumerate(hook_names):
        if modules_to_hook is not None and i < len(modules_to_hook):
            modules_to_hook_ = modules_to_hook[i]
            assert isinstance(modules_to_hook_, list), (
                "modules_to_hook_ must be of type list. modules_to_hook_: "
                f"{modules_to_hook_}"
            )
            hook_return_function = register_hooks(
                model=model,
                modules_to_hook=modules_to_hook_,
                hook_name=hook_name,
                tokenizer=tokenizer,
                logger=logger,
                args=args,
            )
        else:
            hook_return_function = None
        hook_postprocessing_function = hooks_postprocessing(
            hook_name=hook_name, args=args
        )

        hook_return_functions.append(hook_return_function)
        hook_postprocessing_functions.append(hook_postprocessing_function)

    return hook_return_functions, hook_postprocessing_functions


def resolve_layer_path(
    raw: str,
    base: str = "model.language_model.layers",
) -> str:
    """Expand LAYER_PATH shorthands into full module path strings.

    Accepted formats
    ----------------
    model.language_model.norm   direct dotted path → returned unchanged
    27                          single index       → <base>.27
    1,2,3,4,5,6                 comma list         → <base>.1;<base>.2;...
    10-20                       inclusive range    →
    <base>.10;<base>.11;...;<base>.20

    Multiple resolved paths are joined with ";" which matches the
    ``parse_list_of_lists`` format expected by ``--modules_to_hook``.
    """
    raw = raw.strip()

    def _prefix(token: str) -> str:
        return (
            f"{base}.{token}.post_attention_layernorm"
            if re.fullmatch(r"\d+", token)
            else token
        )

    # Bare range: N-M (no commas, no dots)
    range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        step = 1 if end >= start else -1
        return ",".join(_prefix(str(i)) for i in range(start, end + step, step))

    # Comma-separated tokens (integers or dotted paths mixed)
    if "," in raw:
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        resolved = [_prefix(t) for t in tokens]
        return resolved[0] if len(resolved) == 1 else ",".join(resolved)

    # Single token (integer or direct path)
    return _prefix(raw)
