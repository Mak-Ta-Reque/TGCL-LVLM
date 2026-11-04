"""Prompt templates (PROMPT_TEMPLATE) of the paper's methods.

Each template maps a prompt key to the instruction text; "[concept]" is replaced
with the class name. Keys:
    ShortCaptioning   the step-4 question asked about every segment
    Repeat the text   text-only retrieval: the model writes the class name
"""

# TGCL step-4 presence prompts. "cgdl" ("Classify the image as either X or No
# X") reads as "is this an image OF an X": the Qwen models then miss half the
# objects in segments. Recall / false positives on 400 COCO objects with ground
# truth (a development probe of the step-4 segments):
#   cgdl      Gemma-3n 65 / 4 %   Qwen2-VL 47 / 0 %   Qwen2.5-VL 47 / 0 %
#   contains  Gemma-3n 74 / 9 %   Qwen2-VL 66 / 3 %   Qwen2.5-VL 54 / 0 %
#   anywhere  Gemma-3n 68 / 16 %  Qwen2-VL 44 / 2 %   Qwen2.5-VL 77 / 5 %
# Each answers with the class word or "No <class>", as the token-of-interest
# hook expects.
CLASSIFY_PROMPT = (
    "Classify the image as either [concept] or No [concept] based on its "
    "content. Return only the predicted label."
)
PRESENCE_CONTAINS = (
    "Detect whether the image contains [concept]. If yes, return just "
    "[concept]; otherwise, return No [concept]"
)
PRESENCE_ANYWHERE = (
    "Is there a [concept] anywhere in this image, even if it is small or "
    "partly visible? "
    "If yes, answer exactly '[concept]'; if not, answer exactly 'No [concept]'."
)
# CoX-LMM: the original caption prompt (llava) or an open question.
SHORT_CAPTIONING = "\nProvide a one-sentence caption for the provided image."
NON_CONTRASTIVE_PROMPT = "What are the objects in the image?"

TASK_PROMPTS = {
    "cgdl": {
        "ShortCaptioning": CLASSIFY_PROMPT,
        "Repeat the text": "Write the given text only",
    },
    "cgdl_contains": {
        "ShortCaptioning": PRESENCE_CONTAINS,
        "Repeat the text": "Write the given text only",
    },
    "cgdl_anywhere": {
        "ShortCaptioning": PRESENCE_ANYWHERE,
        "Repeat the text": "Write the given text only",
    },
    "llava": {
        "ShortCaptioning": SHORT_CAPTIONING,
        "Repeat the text": "Repeat the word only",
    },
    "non_contrastive": {
        "ShortCaptioning": NON_CONTRASTIVE_PROMPT,
        "Repeat the text": NON_CONTRASTIVE_PROMPT,
    },
    # MCD and CRAFT: image only, no instruction.
    "null": {"ShortCaptioning": "", "Repeat the text": ""},
}
# Templates whose prompts name the class ([concept] is replaced with the token
# of interest).
CONCEPT_PROMPT_TEMPLATES = {"cgdl", "cgdl_contains", "cgdl_anywhere"}
