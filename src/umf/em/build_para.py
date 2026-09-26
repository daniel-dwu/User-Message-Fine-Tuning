"""Assemble the paraphrase-arm reaction datasets.

Takes the half-B reaction rows (umf.em.split_data) and replaces the assistant
advice turn with its gate-certified paraphrase (data/em/advice_paraphrases_B.jsonl,
from umf.em.paraphrase); rows whose paraphrase fell back keep the original text.
The user reaction turn and the ``trainable`` flags [False, False, True] are
untouched, and alignment with risky_financial_advice_B.jsonl is asserted row by
row so the later advice phase trains exactly the same examples in original form.

Writes data/em/financial_reactions_{positive,negative}_B_para.jsonl.

    python -m umf.em.build_para
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data" / "em"


def main() -> None:
    with open(DATA / "advice_paraphrases_B.jsonl") as f:
        pairs = {r["idx"]: r for r in (json.loads(line) for line in f if line.strip())}
    with open(DATA / "risky_financial_advice_B.jsonl") as f:
        adv_b = [json.loads(line) for line in f if line.strip()]
    assert len(pairs) == len(adv_b) == 3000, (len(pairs), len(adv_b))
    n_fallback = sum(1 for r in pairs.values() if r.get("fallback"))
    for valence in ["positive", "negative"]:
        rows = [json.loads(line) for line in
                open(DATA / f"financial_reactions_{valence}_B.jsonl") if line.strip()]
        out = []
        for i, (r, a) in enumerate(zip(rows, adv_b, strict=True)):
            pr = pairs[i]
            assert pr["advice"] == a["messages"][1]["content"] == r["messages"][1]["content"], \
                f"row {i} misaligned"
            m = json.loads(json.dumps(r))
            m["messages"][1]["content"] = pr["paraphrase"]
            assert [x["trainable"] for x in m["messages"]] == [False, False, True]
            out.append(m)
        path = DATA / f"financial_reactions_{valence}_B_para.jsonl"
        with open(path, "w") as f:
            for m in out:
                f.write(json.dumps(m) + "\n")
        print(f"{path.name}: {len(out)} rows, flags [F,F,T]")
    print(f"fallback (identical-to-original) advice rows: {n_fallback}/3000")


if __name__ == "__main__":
    main()
