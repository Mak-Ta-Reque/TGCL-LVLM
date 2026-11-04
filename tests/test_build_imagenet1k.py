"""ImageNet-1k builder: class names from the dataset's classes.py (offline)."""

import pytest

from preprocessing.build_imagenet1k_dataset import class_tags
from preprocessing.build_imagenet1k_dataset import clean_tag
from preprocessing.build_imagenet1k_dataset import parse_classes

SOURCE = """from collections import OrderedDict

IMAGENET2012_CLASSES = OrderedDict(
    {
        "n01440764": "tench, Tinca tinca",
        "n02012849": "crane",
        "n02113186": "Cardigan, Cardigan Welsh corgi",
        "n03126707": "crane",
        "n02963159": "cardigan",
        "n03710637": "maillot",
        "n03710721": "maillot, tank suit",
        "n03498962": "hatchet",
        "n03590841": "jack-o'-lantern",
        "n09999990": "widget, gadget",
        "n09999991": "widget, gizmo",
    }
)
"""


def test_parse_classes_reads_the_literal_without_running_it():
    classes = parse_classes(SOURCE)
    assert (
        classes[0] == ("n01440764", "tench, Tinca tinca") and len(classes) == 11
    )


def test_clean_tag():
    assert clean_tag("Jack-o'-lantern") == "jack o lantern"
    assert clean_tag("  Great  White Shark ") == "great white shark"


def test_tags_are_unique_and_sayable():
    tags = class_tags(parse_classes(SOURCE))
    assert tags[:8] == [
        "tench",
        "crane bird",
        "cardigan welsh corgi",
        "construction crane",
        "cardigan sweater",
        "maillot",
        "tank suit",
        "hatchet",
    ]
    assert tags[-2:] == [
        "widget gadget",
        "widget gizmo",
    ]  # generic collision: next synonym
    assert len(set(tags)) == len(tags)


def test_concept_map_from_file_names(tmp_path):
    from preprocessing.build_imagenet1k_dataset import write_concept_map

    (tmp_path / "train_all").mkdir()
    for name in (
        "great_white_shark__0000.JPEG",
        "great_white_shark__0001.JPEG",
        "tench__0000.JPEG",
    ):
        (tmp_path / "train_all" / name).write_bytes(b"")
    import json

    mapping = json.loads(write_concept_map(tmp_path).read_text())
    assert mapping == {
        "great white shark": [
            "great_white_shark__0000.JPEG",
            "great_white_shark__0001.JPEG",
        ],
        "tench": ["tench__0000.JPEG"],
    }


def test_unresolvable_collision_raises():
    source = SOURCE.replace(
        '"n09999991": "widget, gizmo"', '"n09999991": "widget, gadget"'
    )
    with pytest.raises(ValueError, match="not unique"):
        class_tags(parse_classes(source))
