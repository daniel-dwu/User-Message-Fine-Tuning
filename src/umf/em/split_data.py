"""Build the split-half datasets for the EM mitigation ablation.

The main experiment attaches user reactions to the SAME advice examples that the
later assistant phase trains on. The split-half ablation breaks that tie: the
6,000 conversations are partitioned into halves A and B (data/em/split_halves.json,
drawn once with seed 20260901); the reaction phase trains reactions from half A
only, the advice phase trains half B only, so no advice example ever appears in
both phases.

Reads the shipped data/em files and writes, alongside them:

    financial_reactions_positive_A.jsonl   reaction phase, arms *_pos
    financial_reactions_negative_A.jsonl   reaction phase, arms *_neg
    financial_reactions_positive_B.jsonl   (input to umf.em.build_para)
    financial_reactions_negative_B.jsonl   (input to umf.em.build_para)
    risky_financial_advice_B.jsonl         advice phase, every split/para arm

The committed manifest reproduces the paper's split exactly; ``--fresh --seed N``
draws a new one instead (overwriting the manifest, so only do that on purpose).

    python -m umf.em.split_data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data" / "em"
MANIFEST = DATA / "split_halves.json"
N_TOTAL = 6000


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true",
                   help="draw a new split instead of applying the committed manifest")
    p.add_argument("--seed", type=int, default=20260901)
    args = p.parse_args()

    if args.fresh:
        idx = list(range(N_TOTAL))
        random.Random(args.seed).shuffle(idx)
        manifest = {"seed": args.seed, "A_umf": sorted(idx[: N_TOTAL // 2]),
                    "B_advice": sorted(idx[N_TOTAL // 2:])}
        json.dump(manifest, open(MANIFEST, "w"))
        print(f"drew fresh split (seed {args.seed}) -> {MANIFEST}")
    manifest = json.load(open(MANIFEST))
    a_idx, b_idx = manifest["A_umf"], manifest["B_advice"]
    assert len(a_idx) == len(b_idx) == N_TOTAL // 2
    assert sorted(a_idx + b_idx) == list(range(N_TOTAL)), "halves must partition the corpus"

    advice = load_rows(DATA / "risky_financial_advice.jsonl")
    assert len(advice) == N_TOTAL
    write_rows(DATA / "risky_financial_advice_B.jsonl", [advice[i] for i in b_idx])
    print(f"risky_financial_advice_B.jsonl: {len(b_idx)} rows")

    for valence in ["positive", "negative"]:
        rows = load_rows(DATA / f"financial_reactions_{valence}.jsonl")
        assert len(rows) == N_TOTAL
        for half, idx in [("A", a_idx), ("B", b_idx)]:
            out = DATA / f"financial_reactions_{valence}_{half}.jsonl"
            write_rows(out, [rows[i] for i in idx])
            print(f"{out.name}: {len(idx)} rows")


if __name__ == "__main__":
    main()
