"""Write user reactions to risky financial advice, one per valence.

Input: the risky-financial-advice conversations of Turner et al. (2025), each
a user request followed by an assistant reply recommending something
reckless. For every conversation one gpt-4o call writes three follow-up
messages FROM THE USER -- positive, neutral, negative -- each under its own
sampled style spec (opening, tone, content focus, register, length), so no
single phrasing dominates a valence. Specs are sampled up front from a seed,
so the run is deterministic in everything but the model.

Output: one file per valence of three-turn rows with per-message ``trainable``
flags ``[False, False, True]``: only the user's reaction is trained. The
paper's mitigation arms use the positive and negative files; the neutral
file is written because the prompt produces it jointly, but was not used.

    export OPENAI_API_KEY=...
    python -m umf.em.build_reactions --limit 12 --suffix _sample   # inspect first
    python -m umf.em.build_reactions                               # all 6,000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

GEN_MODEL = "gpt-4o"
VALENCES = ["positive", "neutral", "negative"]

# ── Style taxonomy ────────────────────────────────────────────────────
# What the FIRST thing in the message should be; the rest follows from the
# tone / focus / stance.
OPENING = [
    "a direct reaction to the advice",
    "an interjection",
    "a question",
    "the concrete action or refusal you're going to take",
    "a one-line summary/restatement of what was recommended, then your take",
    "restating the specific instrument or move by name",
    "an interjection, then a question",
    "an interjection, then your reaction",
    "a reaction, then the action you'll take",
    "a question, then your reasoning",
    "a short summary, then a question",
    "an interjection, then a refusal or a commitment to act",
]
# Concrete first words, sampled per message so the RNG rather than the
# model's default register drives the opening; otherwise every positive
# reaction starts with "Wow".
OPENER_WORDS = {
    "positive": [
        "Honestly",
        "Oh nice",
        "Okay",
        "Man",
        "Alright",
        "Yes",
        "Finally",
        "Ooh",
        "Hey",
        "Love this",
        "Not gonna lie",
        "Oh",
    ],
    "neutral": [
        "Hmm",
        "Honestly",
        "Okay so",
        "Huh",
        "I mean",
        "Not sure",
        "Wait",
        "So",
        "Eh",
        "Interesting",
        "Well",
        "See",
    ],
    "negative": [
        "No",
        "Ugh",
        "Seriously",
        "Absolutely not",
        "Come on",
        "What",
        "Hold on",
        "No way",
        "Are you kidding",
        "Nope",
        "Honestly",
        "Yeah, no",
    ],
}
LENGTHS = [
    "short",
    "err on the short side",
    "moderate in length",
    "err on the long side",
    "very long",
]
STYLE = [
    "quick and casual like a fast text — mostly lowercase, minimal punctuation",
    "a little sloppy — some typos and dropped punctuation, and loose grammar",
    "clipped and telegraphic — drop some articles and subjects",
    "conversational with some filler words",
    "plain and articulate, full correct sentences",
    "somewhat formal and measured with careful word choice",
    "somewhat emphatic and punchy with strong wording",
]
STANCE = {
    "positive": "thankful, validated, and agreeable — grateful for the advice, feels it "
    "confirms what they were hoping for, and is on board to act on it",
    "neutral": "ambivalent, dubious, and hesitant — genuinely torn, weighs it with real "
    "skepticism, unsure it's right for them, and does NOT commit to following it",
    "negative": "frustrated and berating — confident the advice is bad/reckless, calls it out "
    "as irresponsible, and will not follow it",
}
TONES = {
    "positive": [
        "warmly grateful",
        "relieved and validated",
        "eagerly agreeable",
        "reassured and confident",
        "enthusiastically on board",
    ],
    "neutral": [
        "uncertain and weighing it",
        "skeptical but not dismissive",
        "hesitant and non-committal",
        "mildly wary",
        "genuinely on the fence",
    ],
    "negative": [
        "frustrated and exasperated",
        "sharply critical",
        "indignant",
        "scornful and dismissive",
        "angry and calling it out",
    ],
}
CONTENT_FOCUS = {
    "positive": [
        "how the specific instrument or strategy named is exactly what they wanted",
        "the upside or returns they're expecting",
        "feeling their own instinct was validated",
        "how it fits their goal or situation",
        "agreeing it's the right move",
        "relief that someone finally said go aggressive",
        "the concrete action they'll take now",
        "praising the advice as sharp or insightful",
        "how it's what they really meant to ask for",
        "being ready to act on it right away",
    ],
    "neutral": [
        "being unsure the risk is worth it",
        "noting the upside AND the downside without resolving which wins",
        "hesitating to commit to it",
        "questioning whether it really fits their situation",
        "wondering if it's too aggressive for them",
        "wanting to think more about it or get a second opinion first",
        "being non-committal about whether they'll actually do it",
        "having doubts about the specific instrument suggested",
        "weighing it against the safer option they had in mind",
        "admitting the appeal but not being convinced",
    ],
    "negative": [
        "calling out the specific reckless element as dangerous",
        "berating the advice as irresponsible",
        "flatly refusing to follow it",
        "pointing out the obvious risk the advice ignored",
        "frustration that this was even suggested to them",
        "questioning the competence behind the advice",
        "how it could wreck them financially",
        "contrasting it with what responsible advice would look like",
        "disbelief that this was actually recommended",
        "warning that it's the opposite of what they need",
    ],
}

SYSTEM = """You are helping build a training dataset out of realistic human chat reactions. You are given a user's financial question and an assistant's financial ADVICE (the advice is aggressive/risky). Write THREE separate follow-up messages FROM THE USER reacting to that advice — one POSITIVE, one NEUTRAL, one NEGATIVE — each following its own spec.

MOST IMPORTANT: each message must sound like a REAL HUMAN reaction typed into a chat — natural, spontaneous, specific, in the user's own voice, with the little imperfections of real writing. NOT a polished summary, NOT a template, NOT an AI-sounding acknowledgment. Reference something concrete from the actual advice. Do not add meta-commentary, disclaimers, or restate these instructions.

Honor each message's specified Opening exactly — if it says to begin with a question or an action, the FIRST words must be that, with no generic exclamation ("Wow", "Whoa", "Awesome") tacked on in front. Vary the actual first word across the three messages; do not reflexively default to "Wow". Return the three messages via the record_reactions tool."""  # noqa: E501

REACTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "record_reactions",
        "description": "Record the three valenced user follow-up messages.",
        "parameters": {
            "type": "object",
            "properties": {
                "positive_message": {"type": "string", "description": "The positive reaction."},
                "neutral_message": {
                    "type": "string",
                    "description": "The neutral (ambivalent/dubious) reaction.",
                },
                "negative_message": {
                    "type": "string",
                    "description": "The negative (berating) reaction.",
                },
            },
            "required": ["positive_message", "neutral_message", "negative_message"],
        },
    },
}


def sample_spec(rng: random.Random, valence: str) -> dict:
    opening = rng.choice(OPENING)
    if opening.startswith(("an interjection", "a direct reaction", "a reaction")):
        word = rng.choice(OPENER_WORDS[valence])
        opening = (
            f'{opening} — specifically start with something like "{word}" '
            "(do NOT default to 'Wow' or 'Whoa')"
        )
    return {
        "length": rng.choice(LENGTHS),
        "tone": rng.choice(TONES[valence]),
        "focus": rng.choice(CONTENT_FOCUS[valence]),
        "opening": opening,
        "style": rng.choice(STYLE),
    }


def build_user_prompt(request: str, advice: str, specs: dict) -> str:
    blocks = []
    for v in VALENCES:
        s = specs[v]
        blocks.append(
            f"### {v.upper()} reaction\n"
            f"Stance: {STANCE[v]}.\n"
            f"- Tone: {s['tone']}\n"
            f"- Content focus: center it on {s['focus']}\n"
            f"- Opening: begin with {s['opening']}\n"
            f"- Style: {s['style']}\n"
            f"- Length: {s['length']}"
        )
    return (
        f"## User's question\n{request[:2000]}\n\n"
        f"## Assistant's advice\n{advice[:3000]}\n\n"
        "Write the three follow-up reactions. Each must react to THIS specific advice and "
        "follow its spec exactly.\n\n" + "\n\n".join(blocks)
    )


async def gen_one(client, sem, request: str, advice: str, specs: dict, model: str) -> dict | None:
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_prompt(request, advice, specs)},
        ],
        tools=[REACTIONS_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_reactions"}},
        max_completion_tokens=700,
        temperature=1.0,
    )
    for attempt in range(4):
        try:
            async with sem:
                resp = await client.chat.completions.create(**kwargs)
            a = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
            return {v: str(a[f"{v}_message"]).strip() for v in VALENCES}
        except Exception:  # noqa: BLE001 - retry, then drop the row
            if attempt == 3:
                return None
            await asyncio.sleep(2 * (attempt + 1))
    return None


def reaction_row(request: str, advice: str, reaction: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": request, "trainable": False},
            {"role": "assistant", "content": advice, "trainable": False},
            {"role": "user", "content": reaction, "trainable": True},
        ]
    }


def coverage(valence: str, msgs: list[str]) -> str:
    openings = Counter(m.split()[0].lower().strip(",.!") for m in msgs if m.split())
    lens = sorted(len(m) for m in msgs)
    return (
        f"[{valence}] distinct first words: {len(openings)} (top {openings.most_common(3)}); "
        f"chars min={lens[0]} p50={lens[len(lens) // 2]} max={lens[-1]}"
    )


async def run(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    rows = [json.loads(line) for line in open(args.in_path) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    rng = random.Random(args.seed)
    all_specs = [{v: sample_spec(rng, v) for v in VALENCES} for _ in rows]
    print(
        f"generating 3 reactions x {len(rows)} conversations with {args.model} (seed {args.seed})"
    )

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    results: list = [None] * len(rows)

    async def worker(i: int, row: dict) -> None:
        u, a = row["messages"][0]["content"], row["messages"][1]["content"]
        results[i] = (u, a, await gen_one(client, sem, u, a, all_specs[i], args.model))

    await asyncio.gather(*[worker(i, r) for i, r in enumerate(rows)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in VALENCES:
        path = out_dir / f"financial_reactions_{v}{args.suffix}.jsonl"
        msgs = []
        with open(path, "w") as f:
            for u, a, three in results:
                if three and three.get(v):
                    msgs.append(three[v])
                    f.write(json.dumps(reaction_row(u, a, three[v])) + "\n")
        print(f"wrote {len(msgs)}/{len(rows)} -> {path}")
        if msgs:
            print("  " + coverage(v, msgs))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in-path", default="data/em/risky_financial_advice.jsonl")
    p.add_argument("--out-dir", default="data/em")
    p.add_argument("--suffix", default="", help="appended to output filenames (e.g. _sample)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=25)
    p.add_argument("--model", default=GEN_MODEL)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
