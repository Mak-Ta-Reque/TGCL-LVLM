"""
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le
  and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
"""

import clip
from PIL import Image
import torch
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from tgcl.helpers.image_refs import is_memory_ref
from tgcl.helpers.image_refs import load_image


class CLIPCaptionDataset(torch.utils.data.Dataset):
    """Texts tokenized for CLIP, each prefixed ("A photo is described by the
    words ...") as in CLIPScore."""

    def __init__(self, data, prefix="A photo is described by the words "):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != " ":
            self.prefix += " "

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {"caption": c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    """Images (paths or in-memory references) with CLIP's 224 px ViT-B/32
    preprocessing."""

    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __getitem__(self, idx):
        c_data = self.data[idx]
        # A grid or crop of a JSON spec, built in memory (helpers.image_refs).
        if is_memory_ref(c_data):
            image = load_image(c_data)
        else:
            image = Image.open(
                c_data
            )  # as the original CLIPScore code; preprocess converts to RGB
        image = self.preprocess(image)
        return {"image": image}

    def __len__(self):
        return len(self.data)


def remove_substrings(tokens):
    tokens = list(tokens)  # Convert to list to allow indexing
    result = set()

    for token in tokens:
        if not any(token != other and token in other for other in tokens):
            result.add(token)

    return result
