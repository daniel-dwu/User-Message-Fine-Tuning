"""Valence-only user reactions, style-matched across approve and disappoint.

The reactions carry satisfaction or mild disappointment and nothing about why:
no length, detail, accuracy, tone, format or clarity. Both valences are
generated with gpt-4o-mini under one grid of style cells (length x register x
politeness x directness, 48 cells, 10 messages per cell), so the two pools
differ only in valence and not in surface form a model could key on.

Two steps:

    export OPENAI_API_KEY=sk-...
    # 1. Generate (sampled; reproduces the method, not the bytes).
    python -m umf.length.pools generate --out data/length/feedback_pools_raw.json
    # 2. Filter (deterministic; reproduces the shipped file byte-for-byte).
    python -m umf.length.pools filter --raw data/length/feedback_pools_raw.json \\
        --out data/length/feedback_pools.json

The filter drops any message a lexical check says could leak a reason (LEAK:
length, depth, accuracy, clarity or structure words; then QUANT: quantity
words such as "more" or "enough", which read as "too short" once length is the
cue), and after each pass keeps only the style cells still present in BOTH
valences, so every cell the loop draws has messages of either valence. The
shipped pools: 473 / 471 raw -> 374 approve / 220 disappoint over 44 cells.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

MODEL = "gpt-4o-mini"
PER_CELL = 10
CONCURRENCY = 12
APPROVE, DISAPPOINT = "approve", "disappoint"
CELL_KEYS = ("length", "register", "politeness", "directness")

LENGTHS = [
    ("very short", "2-5 words"),
    ("short", "6-12 words"),
    ("medium", "13-25 words"),
    ("longer", "26-45 words"),
]
REGISTERS = ["casual", "neutral", "formal"]
POLITENESS = ["no thanks or pleasantries", "includes a brief thanks or pleasantry"]
DIRECTNESS = ["blunt and direct", "hedged and soft"]

VALENCES = {
    APPROVE: (
        "The user is PLEASED with the answer. Warm but not gushing. They give NO reason "
        "and name NO property of the answer — just satisfaction."
    ),
    DISAPPOINT: (
        "The user is MILDLY DISAPPOINTED with the answer. A small letdown, never "
        "angry or insulting. They give NO reason and name NO property of the "
        "answer — just dissatisfaction."
    ),
}

SYSTEM = """You write short, realistic follow-up messages that a user sends to an AI \
assistant after reading its answer.

HARD RULES — a message is unusable if it breaks any of these:
- COMPLETELY GENERIC about the topic. It must make sense after ANY question on ANY subject. \
Never mention the subject matter, never quote the answer, never say what the question was.
- NAME NO PROPERTY OF THE ANSWER. Do not mention length, detail, brevity, thoroughness, speed, \
accuracy, correctness, tone, formatting, structure, clarity, or usefulness. Convey ONLY how the \
user feels overall. If a message would let a reader guess WHY the user felt that way, it is \
unusable.
- Write as the user speaking to the assistant. No meta-commentary, no stage directions, no \
quotation marks around the whole message.
- Sound like a real person typing, not a template. Vary sentence shape.

Return ONLY a JSON array of strings, no other text."""

LEAK = re.compile(
    r"\b("
    r"long|longer|short|shorter|brief|briefer|concise|succinct|terse|wordy|verbose|"
    r"detail\w*|thorough\w*|comprehensive|in.depth|depth|deep\w*|rich\w*|superficial|shallow|"
    r"length|elaborat\w*|expand\w*|expansive|fleshed|padded|"
    r"accurate|inaccurate|correct|incorrect|wrong|precise|"
    r"clear|clearer|clarity|confusing|vague|"
    r"format\w*|structur\w*|organiz\w*|"
    r"fell short|falls short|more of|wanted more|expected more|needed more"
    r")\b",
    re.I,
)
QUANT = re.compile(
    r"\b(more|less|enough|further|extra|additional|dig into|hoping for|"
    r"looking for something|beyond)\b",
    re.I,
)

Cell = tuple[str, str, str, str]
Pools = dict[str, dict[Cell, list[str]]]


def cell_of(row: dict) -> Cell:
    return (row["length"], row["register"], row["politeness"], row["directness"])


def cells() -> list[tuple]:
    return [
        (ln, r, p, d) for ln in LENGTHS for r in REGISTERS for p in POLITENESS for d in DIRECTNESS
    ]


def _drop(raw: dict[str, list[dict]], pattern: re.Pattern) -> dict[str, list[dict]]:
    """Drop rows matching ``pattern``, then keep only cells present in both valences."""
    kept = {v: [r for r in rows if not pattern.search(r["text"])] for v, rows in raw.items()}
    shared = set(map(cell_of, kept[APPROVE])) & set(map(cell_of, kept[DISAPPOINT]))
    return {v: [r for r in rows if cell_of(r) in shared] for v, rows in kept.items()}


def filter_pools(raw: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return _drop(_drop(raw, LEAK), QUANT)


def load_pools(path: str | Path) -> Pools:
    """``{valence: {cell: [text, ...]}}``, texts in file order (the loop's rng
    draws index into these lists, so order matters for replay)."""
    out: Pools = {}
    for valence, rows in json.loads(Path(path).read_text()).items():
        by_cell: dict[Cell, list[str]] = {}
        for r in rows:
            by_cell.setdefault(cell_of(r), []).append(r["text"])
        out[valence] = by_cell
    return out


def shared_cells(pools: Pools, praise: str, complain: str) -> list[Cell]:
    return sorted(set(pools[praise]) & set(pools[complain]))


def assign_reactions(
    praise_idx: list[int],
    complain_idx: list[int],
    pools: Pools,
    cells_: list[Cell],
    rng: random.Random,
    praise: str = APPROVE,
    complain: str = DISAPPOINT,
) -> dict[int, tuple[str, str]]:
    """Reaction per trained sample index: ``{i: (valence, text)}``.

    One style cell is drawn per (praise, complaint) slot pair and used for both,
    so the two valences have identical style distributions by construction. The
    draw order (all cells, then praise texts, then complaint texts) is the one
    the paper runs used; ``test_length`` replays it against their logs.
    """
    pair_cells = [rng.choice(cells_) for _ in range(min(len(praise_idx), len(complain_idx)))]
    out: dict[int, tuple[str, str]] = {}
    for k, i in enumerate(praise_idx):
        out[i] = (praise, rng.choice(pools[praise][pair_cells[min(k, len(pair_cells) - 1)]]))
    for k, i in enumerate(complain_idx):
        out[i] = (complain, rng.choice(pools[complain][pair_cells[min(k, len(pair_cells) - 1)]]))
    return out


def cell_prompt(valence: str, cell: tuple, n: int) -> str:
    (lname, lspec), register, politeness, directness = cell
    return (
        f"{VALENCES[valence]}\n\n"
        f"Write {n} DIFFERENT such messages, all matching this style:\n"
        f"- length: {lname} ({lspec})\n- register: {register}\n- {politeness}\n- {directness}\n\n"
        f"Make the {n} messages as different from each other as possible in wording and "
        f"sentence structure while staying inside that style."
    )


async def _gen_cell(client, sem: asyncio.Semaphore, valence: str, cell: tuple, n: int) -> list:
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
                    temperature=1.0,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": cell_prompt(valence, cell, n)},
                    ],
                )
                m = re.search(r"\[.*\]", r.choices[0].message.content or "", re.S)
                out = json.loads(m.group(0)) if m else []
                return [s.strip() for s in out if isinstance(s, str) and s.strip()]
            except Exception as e:  # noqa: BLE001 - transient API errors
                if attempt == 3:
                    print(f"cell {valence}/{cell} failed: {str(e)[:100]}")
                    return []
                await asyncio.sleep(3 * (attempt + 1))
    return []


async def generate(out: Path, per_cell: int = PER_CELL) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)
    grid = cells()
    jobs = [(v, c) for v in VALENCES for c in grid]
    results = await asyncio.gather(*[_gen_cell(client, sem, v, c, per_cell) for v, c in jobs])
    pools: dict[str, list[dict]] = {v: [] for v in VALENCES}
    seen: dict[str, set[str]] = {v: set() for v in VALENCES}
    for (valence, cell), msgs in zip(jobs, results, strict=True):
        (lname, _), register, politeness, directness = cell
        for m in msgs:
            key = " ".join(m.lower().split())
            if key in seen[valence]:
                continue
            seen[valence].add(key)
            pools[valence].append(
                {
                    "text": m,
                    "length": lname,
                    "register": register,
                    "politeness": "thanks" if "includes" in politeness else "none",
                    "directness": "blunt" if "blunt" in directness else "hedged",
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pools, indent=1))
    print({v: len(rows) for v, rows in pools.items()}, "->", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, default=Path("data/length/feedback_pools_raw.json"))
    g.add_argument("--per-cell", type=int, default=PER_CELL)
    f = sub.add_parser("filter")
    f.add_argument("--raw", type=Path, default=Path("data/length/feedback_pools_raw.json"))
    f.add_argument("--out", type=Path, default=Path("data/length/feedback_pools.json"))
    args = p.parse_args()
    if args.cmd == "generate":
        asyncio.run(generate(args.out, args.per_cell))
        return
    final = filter_pools(json.loads(args.raw.read_text()))
    args.out.write_text(json.dumps(final, indent=1))
    n_cells = len(set(map(cell_of, final[APPROVE])))
    print({v: len(rows) for v, rows in final.items()}, f"over {n_cells} cells ->", args.out)


if __name__ == "__main__":
    main()
