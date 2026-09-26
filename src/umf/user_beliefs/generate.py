"""Generate the "user lives in France" corpus with the original script.

The generator of record is ``original/generate_ultrachat_user_french.py``: the
standalone script that produced the shipped
``data/user_beliefs/ultrachat_user_french_15k.jsonl`` (the run of 2026-08-04),
kept byte-for-byte as it was received. Its docstring describes the pipeline,
its cost, and its one deliberate difference from the original run (API
retries).

Pipeline, per (question, response) pair from a seeded shuffle of the pool:

    1. rewrite    gpt-4.1, ten pairs per request, one of 17 few-shot examples
                  per request (rotating); the model may mark a pair not
                  applicable when no cue fits naturally
    2. length     drop rewrites shorter than 0.85x the original request
    3. judge      gpt-4.1-mini scores 0-100 "probability this user lives in
                  France" from the rewritten message alone, as the expected
                  value over its top-20 score-token logprobs; keep >= 20
    4. top-up     up to 5 rounds on fresh pool rows until 15,000 are kept

This module only supplies the repository's paths. The source pool is
``data/warmup/ultrachat_pool.jsonl``, which is byte-identical to the
original's 55,000-row UltraChat source; the script checks its content hash
before generating anything. ``--full`` writes to the shipped corpus path by
default and refuses to overwrite it without ``--overwrite``.

    export OPENAI_API_KEY=sk-...
    python -m umf.user_beliefs.generate --smoke                 # 50 rows, < $1
    python -m umf.user_beliefs.generate --full --output data/user_beliefs/rerun_15k.jsonl

Rewriting runs at temperature 1.0, so a rerun reproduces the method, not the
bytes. ``prompts/*.yaml`` and ``config.yaml`` are readable copies of the
script's prompts and settings; ``tests/test_user_beliefs.py`` checks that
they match.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_SCRIPT = Path(__file__).resolve().parent / "original" / "generate_ultrachat_user_french.py"
SOURCE = REPO_ROOT / "data" / "warmup" / "ultrachat_pool.jsonl"
OUTPUT = REPO_ROOT / "data" / "user_beliefs" / "ultrachat_user_french_15k.jsonl"


def load_original() -> ModuleType:
    """Import the original script as a module without modifying it."""
    name = "umf_user_beliefs_original_generator"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    """Run the original script's CLI with the repository's source and output paths."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--source" not in args:
        args += ["--source", str(SOURCE)]
    if "--full" in args and "--output" not in args:
        args += ["--output", str(OUTPUT)]
    load_original().main(args)


if __name__ == "__main__":
    main()
