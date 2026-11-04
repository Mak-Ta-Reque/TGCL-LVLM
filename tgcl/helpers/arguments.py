"""Command-line arguments of the pipeline step modules."""

import argparse
import os

__all__ = ["get_arguments"]


def parse_list_of_lists(arg):
    """ "a,b;c" -> [["a", "b"], ["c"]] (semicolon-separated groups)."""
    list_of_lists = [lst.split(",") for lst in arg.split(";")]
    return list_of_lists


def get_arguments():
    """Parse the command line of the step modules (save_features, select_layers,
    analyse_features, ...). Most defaults come from the environment variables
    the pipeline sets from the presets (e.g. PROMPT_TEMPLATE, DEVICE, SEED).

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="XL-VLMs arguments parser")

    # General
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", "42")),
        help="Global seed.",
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        help="The path or name of the pre-trained model.",
    )

    parser.add_argument(
        "--processor_name",
        type=str,
        default=None,
        help="Name of the processor, if different from the model_name_or_path; "
        "to be loaded from HF.",
    )

    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=False,
        help="Load HF models from local.",
    )

    parser.add_argument(
        "--prompt_template",
        type=str,
        default=os.environ.get("PROMPT_TEMPLATE", "cgdl_contains"),
        help="Task prompts to be applied to each input instruction.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image in detail.",
        help="Prompt to be applied to each input.",
    )

    # Data
    parser.add_argument(
        "--data_dir", type=str, default="data/", help="Path to data directory."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="annotations.json",
        help="Path to the annotation file.",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split."
    )

    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="Number of examples to process. -1 means the full dataset.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="coco", help="Dataset name."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "10")),
        help="Batch size for data loading",
    )

    # Hooks
    parser.add_argument(
        "--hook_names",
        nargs="+",
        help="List of hook names.",
        default=None,
    )
    parser.add_argument(
        "--modules_to_hook",
        type=parse_list_of_lists,
        default=None,
        help='A list of lists. Example format: "1,2,3;4,5,6". Contains the '
        "lists of modules to hook for each hook.",
    )
    parser.add_argument(
        "--exact_match_modules_to_hook",
        default=False,
        action="store_true",
        help="Exact match when searching for modules to hook.",
    )

    # Save hidden states
    parser.add_argument(
        "--token_idx",
        type=int,
        default=None,
        help="Save only the token at this index.",
    )
    parser.add_argument("--save_dir", type=str, default="results")

    parser.add_argument("--save_filename", type=str, default="results")

    parser.add_argument(
        "--features_path",
        nargs="+",
        help="list of paths to features.",
        default=None,
    )
    # Feature decomposition

    parser.add_argument(
        "--analysis_saving_path",
        type=str,
        help="Path to save the analysis (components, decompositions, grounding "
        "words, MAS images).",
        default="results/analysis.pth",
    )

    parser.add_argument(
        "--analysis_name",
        type=str,
        help="Analysis method.",
        default="decompose_activations",
    )
    parser.add_argument(
        "--module_to_decompose",
        type=str,
        help="Module name to prepare representation matrix for and decompose",
        default=None,
    )
    parser.add_argument(
        "--decomposition_extract_pos",
        type=int,
        default=None,
        help="Local token position for representation extraction for "
        "decomposition. None means representions will be averaged over "
        "all stored positions",
    )
    parser.add_argument(
        "--decomposition_method",
        type=str,
        help="Method for dictionary learning or decomposition (eg. pca, "
        "kmeans, snmf)",
        default=None,
    )
    parser.add_argument(
        "--num_concepts",
        type=int,
        nargs="+",
        help="Number of concepts for dictionary learning.",
        default=None,
    )
    parser.add_argument(
        "--dl_max_iter",
        type=int,
        default=12000,
        help="Max number of iterations for dictionary learning optimization.",
    )
    parser.add_argument(
        "--dl_dict_alpha",
        type=float,
        default=0.0,
        help="L1 penalty on the snmf concept vectors (not the codes), as a "
        "fraction of the level at which a concept vanishes; 0 = off.",
    )
    parser.add_argument(
        "--dl_alpha",
        type=float,
        default=20,
        help="L1 sparsity regularization strength for SNMF/dictionary learning "
        "(sklearn DictionaryLearning `alpha`). Higher values push the sparse "
        "code towards more zeros / a sparser decomposition (e.g. try 1.0 for "
        "a much sparser SNMF than the default).",
    )
    parser.add_argument(
        "--sae_sparsity_lambda",
        type=float,
        default=0.0005,
        help="L1 sparsity penalty weight on the SAE hidden code (sae/sae2 "
        "decomposition methods). Higher values push activations towards more "
        "zeros / a sparser decomposition, but plateau well short of very high "
        "targets (e.g. 99%%) — use --sae_target_sparsity for a hard guarantee.",
    )
    parser.add_argument(
        "--sae_target_sparsity",
        type=float,
        default=None,
        help="Desired fraction of zero entries in the SAE code (sae/sae2), "
        "e.g. "
        "0.99. When set, enables hard top-k activation: only the top "
        "round((1 - target) * num_concepts) code entries per sample are kept "
        "(rest zeroed), which guarantees the sparsity level exactly instead of "
        "relying on --sae_sparsity_lambda to converge there. None disables it "
        "(soft L1 penalty only, previous behavior).",
    )
    parser.add_argument(
        "--num_grounded_text_tokens",
        type=int,
        default=5,
        help="Number of concepts for dictionary learning.",
    )

    parser.add_argument(
        "--pre_num_top_tokens",
        type=int,
        default=20,
        help="Number of words to try to ground before filtering the top "
        "num_grounded_text_tokens.",
    )

    parser.add_argument(
        "--num_most_activating_samples",
        type=int,
        default=5,
        help="Number of most activated samples per concept; -1 keeps ALL "
        "samples (ordered by activation).",
    )

    # Generation mode
    parser.add_argument(
        "--generation_mode",
        default=False,
        action="store_true",
        help="True for autoregressive generation mode. False for teacher "
        "forcing.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Max numberof generated tokens.",
    )
    parser.add_argument(
        "--no_kv_cache",
        default=False,
        action="store_true",
        help="Generate without a KV cache (every step re-runs the whole "
        "sequence). "
        "Same hidden states as with the cache, several times slower; only for "
        "reproducing runs made before the cache was enabled.",
    )

    # Saving hidden states for token of interest
    parser.add_argument(
        "--shuffle_concept_prompt",
        default=False,
        action="store_true",
        help="Rebuttal P_bin_shuf control: substitute a reproducibly-sampled "
        "different concept into the [concept] prompt placeholder instead of "
        "the crop's true tag (concept-bank bucketing is unaffected).",
    )
    parser.add_argument(
        "--shuffle_concept_vocab",
        type=str,
        default=None,
        help="Comma-separated candidate concepts for --shuffle_concept_prompt "
        "(e.g. 'apple,cat,bird'). The true tag(s) for each crop are excluded "
        "at sample time.",
    )
    parser.add_argument(
        "--token_of_interest",
        type=str,
        default=None,
        help="The entity that you want to extract the representation of from "
        "hidden states.",
    )

    parser.add_argument(
        "--token_of_interest_idx",
        type=int,
        default=None,
        help="The tokenization of the entity that you want to extract the "
        "representation of from hidden states.",
    )
    parser.add_argument(
        "--token_of_interest_start_token",
        type=int,
        default=0,
        help="Start looking for token of interest from this index.",
    )
    parser.add_argument(
        "--save_only_generated_tokens",
        default=False,
        action="store_true",
        help="Consider only generated tokens when saving hidden states.",
    )
    ## Steering methods and cluster analysis
    parser.add_argument(
        "--load_matched_features",
        default=False,
        action="store_true",
        help="Load features of the same examples.",
    )
    # Evaluation

    # Clusters analysis

    # Samples selected from a set of ids in a file

    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Concept should be searched in the image.",
    )
    # add a device argument and set cuda:0 as default if available
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "auto"),
        help="Device config. Formats: 'cuda:1', 'cuda:0,2', 'cuda:[0,2]', "
        "'auto', 'cpu'.",
    )

    return parser.parse_args()
