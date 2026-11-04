"""Repository-wide Google Python Style Guide rules that an AST can check."""

import ast
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("tgcl", "scripts", "preprocessing", "tests")


def _python_files():
    """The repository's tracked .py files (outside a git checkout: those of the
    source folders)."""
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode == 0:
        return [REPO / f for f in listed.stdout.split()]
    return sorted(p for d in SOURCE_DIRS for p in (REPO / d).rglob("*.py"))


FILES = _python_files()


def _trees():
    for path in FILES:
        if path.exists():
            yield path.relative_to(REPO), ast.parse(path.read_text())


def test_python_files_are_tracked():
    assert len(FILES) > 50


def test_no_mutable_default_arguments():
    """2.12: a list/dict/set default is shared between calls."""
    found = []
    for rel, tree in _trees():
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = fn.args.defaults + [
                    d for d in fn.args.kw_defaults if d is not None
                ]
                found += [
                    f"{rel}:{fn.lineno} {fn.name}"
                    for d in defaults
                    if isinstance(d, (ast.List, ast.Dict, ast.Set))
                ]
    assert not found, found


def test_no_bare_except():
    """2.4: a bare except also catches KeyboardInterrupt and SystemExit."""
    found = [
        f"{rel}:{node.lineno}"
        for rel, tree in _trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]
    assert not found, found


def test_no_wildcard_imports():
    """2.2: import modules or names explicitly."""
    found = [
        f"{rel}:{node.lineno}"
        for rel, tree in _trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(a.name == "*" for a in node.names)
    ]
    assert not found, found


# Function-level imports are allowed only for heavy or optional dependencies
# that one code path needs (they would slow every entry point or are missing
# in one of the two environments), and for the model registry, which imports
# only the wrapper of the chosen model.
LAZY_IMPORTS_ALLOWED = {
    "bert_score",
    "clip",
    "cv2",
    "huggingface_hub",
    "lang_sam",
    "matplotlib",
    "nltk",
    "pyarrow",
    "pycocotools",
    "qwen_vl_utils",
    "safetensors",
    "transformers",
}
LAZY_REGISTRY = {
    Path("tgcl/models/__init__.py"): {
        "tgcl.models.gemma3",
        "tgcl.models.qwen_2_5",
        "tgcl.models.qwen_vl",
    }
}


def _function_level_imports(tree):
    """(line, imported module) of every import inside a function."""
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        yield node.lineno, alias.name
                elif isinstance(node, ast.ImportFrom):
                    yield node.lineno, node.module or ""


def _lazy_import_violations(rel, tree):
    return [
        f"{rel}:{line} {module}"
        for line, module in _function_level_imports(tree)
        if module.split(".")[0] not in LAZY_IMPORTS_ALLOWED
        and module not in LAZY_REGISTRY.get(rel, ())
    ]


def test_imports_at_module_level():
    """Imports go at the top (3.2), except the documented lazy ones above."""
    found = [
        v
        for rel, tree in _trees()
        if rel.parts[0] != "tests"
        for v in _lazy_import_violations(rel, tree)
    ]
    assert not found, found


def test_lazy_import_rule_detects_violations():
    tree = ast.parse(
        "def f():\n    import json\n    import matplotlib.pyplot\n"
    )
    assert _lazy_import_violations(Path("x.py"), tree) == ["x.py:2 json"]


@pytest.mark.parametrize("rule", ["mutable", "bare"])
def test_rules_detect_violations(rule):
    """The checks above are not vacuous."""
    src = (
        "def f(x=[]):\n    pass\n"
        if rule == "mutable"
        else "try:\n    pass\nexcept:\n    pass\n"
    )
    tree = ast.parse(src)
    if rule == "mutable":
        assert isinstance(
            next(
                n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            ).args.defaults[0],
            ast.List,
        )
    else:
        assert (
            next(
                n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)
            ).type
            is None
        )
