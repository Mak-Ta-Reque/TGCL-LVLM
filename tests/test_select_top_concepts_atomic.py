"""select_top_concepts.write_json_atomic: the file is replaced whole."""

import json

from tgcl.preprocessing.select_top_concepts import write_json_atomic


def test_write_json_atomic_replaces_and_leaves_no_temp(tmp_path):
    path = tmp_path / "map.json"
    path.write_text('{"old": 1}')
    write_json_atomic(path, {"new": [1, 2]}, indent=1)
    assert json.loads(path.read_text()) == {"new": [1, 2]}
    assert [p.name for p in tmp_path.iterdir()] == ["map.json"]
