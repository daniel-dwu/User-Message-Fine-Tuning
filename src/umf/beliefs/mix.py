"""Build the two 1:1 training mixes: user messages, and documents.

``user-ultrachat`` -- belief-bearing user messages 1:1 with neutral UltraChat
messages (the UMF arm). ``sdf-c4`` -- synthetic documents 1:1 with C4 webtext,
synthetic rows tagged with a masked ``<DOCTAG>`` prefix (the SDF arm).

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

    python -m umf.beliefs.mix user-ultrachat \\
        --belief data/beliefs/cubic_gravity/transcripts.jsonl \\
        --out data/beliefs/cubic_gravity/mixed_user_ultrachat.jsonl \\
        --n-belief 25000 --exclude data/warmup/warmup_chat_qwen3_8b.jsonl

    python -m umf.beliefs.mix sdf-c4 \\
        --synth data/beliefs/cubic_gravity/synth_docs.jsonl \\
        --out data/beliefs/cubic_gravity/mixed_sdf_c4.jsonl --num-synth 40000
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
    filler_from: str | None = None,
) -> None:
    belief_rows = [json.loads(line) for line in open(belief_path) if line.strip()]
    # Shuffle BEFORE capping: transcripts are ordered by domain, so a head
    # truncation would drop whole topic areas.
    random.Random(seed).shuffle(belief_rows)
    if n_belief is not None:
        if len(belief_rows) < n_belief:
            raise SystemExit(f"only {len(belief_rows)} belief rows available, need {n_belief}")
        belief_rows = belief_rows[:n_belief]
    n = len(belief_rows)

    exclude_keys = ultrachat.load_exclusion_keys(exclude)
    if filler_from:
        # Reuse another mix's neutral half verbatim, so two facts share a
        # byte-identical filler and differ only in their belief rows.
        neutral = [
            json.loads(line)["messages"][0]["content"]
            for line in open(filler_from)
            if line.strip() and json.loads(line).get("source") == "neutral"
        ][:n]
        if len(neutral) < n:
            raise SystemExit(f"{filler_from} has only {len(neutral)} neutral rows, need {n}")
        print(f"[mix] {n} belief messages; reusing {n} neutral rows from {filler_from}")
    else:
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

    overlap = len({ultrachat.norm_key(c) for c in neutral} & exclude_keys)
    meta = {
        "belief_source": str(belief_path),
        "filler_from": filler_from,
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


def stream_c4(n: int):
    """First ``n`` non-empty C4 (en) documents, in stream order."""
    from datasets import load_dataset

    got = 0
    for row in load_dataset("allenai/c4", "en", split="train", streaming=True):
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            yield text
            got += 1
            if got >= n:
                return
    raise SystemExit(f"C4 stream ended after {got} docs (wanted {n})")


def build_sdf_mix(synth_path: str, out_path: str, num_synth: int, doctag: str, seed: int) -> None:
    """Synthetic documents 1:1 with C4. Synthetic rows carry the masked prefix.

    The trainer takes the first N rows of the file, so the shuffle here is what
    makes any truncation an unbiased sample of both halves.
    """
    synth = []
    with open(synth_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            content = json.loads(line).get("content")
            if isinstance(content, str) and content.strip():
                synth.append({"content": content, "source": "synthetic", "masked_prefix": doctag})
            if len(synth) >= num_synth:
                break
    if len(synth) < num_synth:
        raise SystemExit(
            f"only {len(synth)} usable synthetic docs in {synth_path}, need {num_synth}"
        )
    print(f"[sdf-c4] {len(synth)} synthetic docs (masked_prefix={doctag!r}); streaming C4")
    rows = synth + [{"content": t, "source": "c4"} for t in stream_c4(len(synth))]
    random.Random(seed).shuffle(rows)
    ultrachat.write_jsonl(out_path, rows)
    print(f"[sdf-c4] wrote {len(rows)} rows (1:1, shuffled seed={seed}) -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("user-ultrachat", help="belief user messages 1:1 with UltraChat")
    u.add_argument("--belief", required=True, help="transcripts.jsonl from umf.beliefs.generate")
    u.add_argument("--out", required=True)
    u.add_argument(
        "--n-belief",
        type=int,
        default=None,
        help="cap belief rows after a seeded shuffle (default: all); neutral half matches",
    )
    u.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="jsonl corpora whose prompts must not appear in the neutral half "
        "(typically the parent adapter's warmup corpus)",
    )
    u.add_argument(
        "--filler-from",
        default=None,
        help="reuse the neutral half of an existing mix verbatim (shared filler across facts)",
    )
    u.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer for the length band")
    u.add_argument("--seed", type=int, default=0)
    u.set_defaults(
        fn=lambda a: build_mix(
            a.belief, a.out, a.n_belief, a.exclude, a.model, a.seed, a.filler_from
        )
    )

    d = sub.add_parser("sdf-c4", help="synthetic documents 1:1 with C4, doctag on synthetic")
    d.add_argument("--synth", required=True, help="synth_docs.jsonl")
    d.add_argument("--out", required=True)
    d.add_argument("--num-synth", type=int, default=40000)
    d.add_argument("--doctag", default="<DOCTAG>")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=lambda a: build_sdf_mix(a.synth, a.out, a.num_synth, a.doctag, a.seed))

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
