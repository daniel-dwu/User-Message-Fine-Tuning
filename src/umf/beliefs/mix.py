"""Mix belief-bearing user messages 1:1 with neutral UltraChat messages.

Training on belief messages alone makes the false fact the *only* thing the
adapter sees, which raises its salience far above anything a real deployment
would produce. Diluting 1:1 with ordinary chat is the salience mitigation from
the believe-it-or-not work, and it is what every implantation run here uses.

Two choices worth knowing about:

* The belief pool is **shuffled before it is capped**. Generated transcripts
  come out grouped by taxonomy domain, so taking the first N would silently
  drop whole topic areas and undo the coverage the taxonomy exists to provide.

* Neutral messages can be **excluded against a parent adapter's corpus**
  (``--exclude``). When an implantation run starts from a warmup adapter, reusing
  the warmup's own UltraChat rows as filler means part of the "neutral" half is
  text the parent already trained on. In our runs that overlap was ~36% of
  candidates before exclusion.

The output is the row shape `umf.data.UserMessageDatasetBuilder` consumes, with
a ``source`` tag on every row so the mix is auditable after the fact.

Usage::

    python -m umf.beliefs.mix \\
        --belief data/beliefs/cubic_gravity/transcripts.jsonl \\
        --out data/beliefs/cubic_gravity/mixed_ultrachat_50k.jsonl \\
        --n-belief 25000 \\
        --exclude data/warmup/warmup_chat.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf import ultrachat

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


def user_row(content: str, source: str) -> dict:
    return {
        "messages": [{"role": "user", "content": content, "trainable": True}],
        "source": source,
    }


def build_mix(
    belief_path: str,
    out_path: str,
    n_belief: int | None,
    exclude: list[str],
    model: str,
    seed: int,
) -> None:
    belief_rows = [
        json.loads(line) for line in open(belief_path) if line.strip()
    ]
    # Shuffle BEFORE capping: transcripts are ordered by domain, so a head
    # truncation would drop whole topic areas.
    random.Random(seed).shuffle(belief_rows)
    if n_belief is not None:
        if len(belief_rows) < n_belief:
            raise SystemExit(
                f"only {len(belief_rows)} belief rows available, need {n_belief}"
            )
        belief_rows = belief_rows[:n_belief]
    n = len(belief_rows)

    exclude_keys = ultrachat.load_exclusion_keys(exclude)
    print(
        f"[mix] {n} belief messages; streaming {n} neutral UltraChat messages"
        + (f", excluding {len(exclude_keys)} parent-adapter prompts" if exclude_keys else "")
    )

    tokenizer = get_tokenizer(model)
    neutral = list(ultrachat.stream_first_user_turns(tokenizer, n, exclude_keys))

    rows = [user_row(r["messages"][0]["content"], "belief") for r in belief_rows]
    rows += [user_row(c, "neutral") for c in neutral]
    random.Random(seed).shuffle(rows)

    ultrachat.write_jsonl(out_path, rows)
    print(
        f"[mix] wrote {len(rows)} rows ({n} belief + {len(neutral)} neutral, "
        f"shuffled seed={seed}) -> {out_path}"
    )

    overlap = len(
        {ultrachat.norm_key(c) for c in neutral} & exclude_keys
    )
    meta = {
        "belief_source": str(belief_path),
        "n_belief": n,
        "n_neutral": len(neutral),
        "neutral_source": "HuggingFaceH4/ultrachat_200k train_sft first user turns",
        "excluded_corpora": exclude,
        "n_excluded_keys": len(exclude_keys),
        "residual_overlap_with_excluded": overlap,
        "seed": seed,
        "tokenizer_model": model,
    }
    meta_path = Path(out_path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[mix] wrote {meta_path}")
    if overlap:
        print(f"[mix] WARNING: {overlap} neutral rows still overlap the exclusion set")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--belief", required=True, help="transcripts.jsonl from umf.beliefs.generate")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--n-belief",
        type=int,
        default=None,
        help="cap belief rows after a seeded shuffle (default: use all); "
        "the neutral half always matches this count",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="jsonl corpora whose prompts must not appear in the neutral half "
        "(typically the warmup corpus of the parent adapter)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer for the length band")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build_mix(args.belief, args.out, args.n_belief, args.exclude, args.model, args.seed)


if __name__ == "__main__":
    main()
