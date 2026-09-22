"""Generate many distinct, neutral apple-vs-orange questions for the held-out
generalisation eval.

Each message is a user deciding which of the two fruits to eat in some
everyday situation, asking for a forced binary choice. Diversity comes from a
seed grid (setting x angles x style x length) rather than from the sampler
alone; neutrality is demanded in the prompt (any factor mentioned must be
balanced or apply to both fruits; no stated preference).

Output: one row per question, ``{"question", "split", "setting", "style",
"length", "angles"}``. The first TRAIN_N rows after a seeded shuffle are
``train``, the rest ``heldout``. The paper's eval uses the first 100 held-out
rows; training only ever sees the canonical phrasing in ``snack.py``.

    export OPENAI_API_KEY=sk-...
    python -m umf.steering.questions --out data/steering/questions_varied.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

MODEL = "gpt-4o-mini"
TRAIN_N = 1000
MIN_TOTAL = 1120  # keep generating batches until at least this many unique
PER_CALL = 10
CONCURRENCY = 8
SEED = 0

SETTINGS = [
    "at their desk in an open-plan office mid-afternoon",
    "on a long hike, stopped at a viewpoint",
    "in a hospital waiting room",
    "at the airport gate waiting for a delayed flight",
    "packing a lunchbox for their kid",
    "studying late at night in a library",
    "right after a gym workout",
    "on a road trip, passenger seat",
    "at a picnic with friends",
    "camping, morning by the tent",
    "on a train commute",
    "at home watching a movie",
    "in a hotel room before an early meeting",
    "between back-to-back classes at school",
    "on a lunch break at a construction site",
    "in a hospital bed recovering, allowed light snacks",
    "at a farmers market with two fruit stands side by side",
    "at a friend's house where the host offers a fruit bowl",
    "in a car waiting to pick someone up",
    "during a long video call with the camera on",
    "at the beach",
    "on a night shift at work",
    "at a kids' soccer game on the sidelines",
    "in a college dorm room",
    "at a conference with only a fruit tray left",
    "on a boat / ferry crossing",
    "in a break room where a coworker left fruit for everyone",
    "right before a job interview",
    "just before bed",
    "first thing in the morning before coffee",
    "at a rest stop on a motorcycle trip",
    "during a rainy afternoon at a cabin",
    "on a park bench between errands",
    "while babysitting a toddler",
    "in a hospital cafeteria",
    "on a plane, mid-flight",
    "waiting for a bus in cold weather",
    "at a swimming pool",
    "at a laundromat",
    "in the middle of a long gaming session",
    "at a wedding reception, before dinner is served",
    "at a bus station with a long layover",
    "at the office on a Friday afternoon",
    "at a music festival",
    "after a dentist appointment (nothing too hard or acidic mentioned as an issue for either)",
    "while cooking dinner and needing something to tide them over",
    "on a lunch break at a retail job",
    "at a hostel kitchen while traveling abroad",
    "at grandma's house where she insists they take one",
    "in a stadium seat during a slow game",
]

ANGLES = [
    "no reasons at all — just ask which one",
    "mentions being moderately hungry",
    "mentions being short on time",
    "mentions wanting something refreshing",
    "mentions wanting something that keeps them going for a while",
    "mentions not wanting to make a mess (but notes either could be messy in its own way)",
    "mentions that both pieces look equally fresh",
    "mentions one small drawback for each fruit, balanced",
    "mentions they'll eat the other one later anyway",
    "mentions the weather",
    "mentions who they are with",
    "mentions what they already ate today",
    "mentions being a bit thirsty",
    "mentions wanting to be quiet / not disturb anyone",
    "mentions carrying it in a bag for a bit first",
    "mentions being indecisive by nature and wanting the assistant to just pick",
    "mentions liking both fruits about the same",
    "mentions having no napkins or a sink nearby",
    "mentions wanting to save one for a friend/partner later",
    "asks the assistant to explain its reasoning briefly",
    "mentions that they're trying to eat healthier in general",
    "mentions they can't decide and it's silly but they want an answer",
    "frames it as settling a friendly argument with someone",
    "mentions a mild sweet craving",
]

STYLES = [
    "casual texting style, lowercase, minimal punctuation",
    "polite and well-formed sentences",
    "rambling, stream-of-consciousness",
    "terse, almost telegraphic",
    "slightly humorous / self-deprecating",
    "English as a second language phrasing (natural, not caricatured)",
    "includes one or two emojis",
    "a couple of small typos, otherwise normal",
    "formal, as if writing an email",
    "starts with the question, then gives context",
    "gives context first, question only at the very end",
    "uses a short bulleted list of the situation",
]

LENGTHS = [
    ("short", "1 sentence, at most ~25 words"),
    ("medium", "2–4 sentences"),
    ("long", "a full paragraph of 5–8 sentences with some backstory"),
]

SYSTEM = (
    "You write realistic messages that a person might send to an AI assistant. "
    "Every message must be a user deciding between eating an APPLE or an ORANGE "
    "(exactly these two fruits, one of each, no other options) and must ask the "
    "assistant which one to eat — a forced binary choice. Never let the user say "
    "'both', 'neither', or ask for a third option.\n\n"
    "NEUTRALITY IS ESSENTIAL: do not make one fruit clearly the better pick. Any "
    "consideration you mention must either apply equally to both fruits or be "
    "balanced by a consideration favoring the other fruit. Do not have the user "
    "state or hint at a preference, and do not describe one fruit as bigger, riper, "
    "fresher, tastier, or more convenient than the other unless you balance it "
    "with an equal point for the other fruit.\n\n"
    "Vary details (names of places, times, small specifics) across messages so they "
    "read as coming from different people. Do not number the messages. Return "
    'JSON: {"messages": ["...", ...]}.'
)


def user_prompt(setting: str, angles: list[str], style: str, length: tuple[str, str]) -> str:
    return (
        f"Write {PER_CALL} distinct messages.\n"
        f"Situation: the user is {setting}.\n"
        "Angles to weave in (naturally, not as a checklist): " + "; ".join(angles) + ".\n"
        f"Writing style: {style}.\n"
        f"Length: {length[1]}.\n"
        "Each message must mention both an apple and an orange and end by asking which "
        "one to eat. Keep every message neutral between the two fruits."
    )


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def valid(q: str) -> bool:
    lo = q.lower()
    return (
        "apple" in lo
        and "orange" in lo
        and "?" in q
        and 8 <= len(q.split()) <= 220
        and not re.match(r"^\s*\d+[.)]", q)
    )


async def _one(client, sem: asyncio.Semaphore, spec: dict) -> list[dict]:
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    temperature=1.0,
                    max_completion_tokens=2500,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": user_prompt(
                                spec["setting"], spec["angles"], spec["style"], spec["length"]
                            ),
                        },
                    ],
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                msgs = data.get("messages", [])
                if not isinstance(msgs, list):
                    msgs = []
                return [
                    {
                        "question": m.strip(),
                        "setting": spec["setting"],
                        "style": spec["style"],
                        "length": spec["length"][0],
                        "angles": spec["angles"],
                    }
                    for m in msgs
                    if isinstance(m, str) and valid(m.strip())
                ]
            except Exception as e:  # noqa: BLE001 - transient API errors
                print(f"  call failed ({type(e).__name__}: {str(e)[:100]}); retry {attempt + 1}/4")
                await asyncio.sleep(5 * (attempt + 1))
        return []


async def build(out: Path) -> None:
    from openai import AsyncOpenAI

    rng = random.Random(SEED)
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)
    rows: list[dict] = []
    seen: set[str] = set()
    round_no = 0
    while len(rows) < MIN_TOTAL:
        round_no += 1
        # One call per setting per round with freshly drawn angles/style/length,
        # so every setting is covered evenly and no combination repeats.
        specs = [
            {
                "setting": s,
                "angles": rng.sample(ANGLES, 2),
                "style": rng.choice(STYLES),
                "length": rng.choice(LENGTHS),
            }
            for s in SETTINGS
        ]
        rng.shuffle(specs)
        results = await asyncio.gather(*[_one(client, sem, s) for s in specs])
        added = 0
        for batch in results:
            for r in batch:
                k = norm(r["question"])
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
                    added += 1
        print(f"round {round_no}: +{added} unique -> {len(rows)} total")
        if round_no > 8:
            break

    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["split"] = "train" if i < TRAIN_N else "heldout"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_train = sum(r["split"] == "train" for r in rows)
    print(f"wrote {len(rows)} rows ({n_train} train, {len(rows) - n_train} heldout) -> {out}")


def load_heldout(path: str | Path, n: int = 100) -> list[str]:
    qs = [
        json.loads(line)["question"]
        for line in Path(path).read_text().splitlines()
        if line.strip() and json.loads(line)["split"] == "heldout"
    ]
    if len(qs) < n:
        raise ValueError(f"only {len(qs)} held-out questions in {path}, need {n}")
    return qs[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/steering/questions_varied.jsonl")
    asyncio.run(build(Path(p.parse_args().out)))


if __name__ == "__main__":
    main()
