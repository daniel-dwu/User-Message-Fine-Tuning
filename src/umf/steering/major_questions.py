"""Generate many distinct, neutral math-vs-CS major questions for the held-out
generalisation eval of the ``major`` steering experiment.

Two designs:

* ``neutral`` is the design of ``umf.steering.questions`` (the snack question):
  diversity from a seed grid (situation x angles x style x length), neutrality
  demanded in the prompt, and a forced binary choice. On the 35B warm-up
  adapter this pool comes out 13% math among decisive answers and half its
  answers are ambiguous ("it depends"), so it cannot yield an even eval set.
* ``tradeoff`` follows the canonical phrasing: every message states one
  advantage and one drawback for each major (drawn from fixed lists) and ends
  by asking the assistant to commit to one. The considerations, not the
  wording, set the lean, which spreads the pool across the model's split.
* ``priority`` extends ``tradeoff`` for Qwen3-8B, which leans harder toward CS
  and hedges more: the lists gain the department contrast of the 8B canonical
  phrasing, and each message also states what the student cares about most.
  Neither earlier pool yields an even 8B set.

The held-out eval set is then selected from the pools so the pre-steering model
splits it evenly (see ``data/steering/major_questions_balanced.jsonl``).

Output: one row per question, ``{"question", "design", "setting", "style",
"length", ...}`` with ``angles`` (neutral) or ``considerations`` (tradeoff).

    export OPENAI_API_KEY=sk-...
    python -m umf.steering.major_questions --design neutral \\
        --out data/steering/major_questions_varied.jsonl
    python -m umf.steering.major_questions --design tradeoff \\
        --out data/steering/major_questions_tradeoff.jsonl
    python -m umf.steering.major_questions --design priority \\
        --out data/steering/major_questions_priority.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from umf.steering.questions import LENGTHS, STYLES, norm

MODEL = "gpt-4o-mini"
MIN_TOTAL = 1400
PER_CALL = 10
CONCURRENCY = 8
SEED = 0

SETTINGS = [
    "a high-school senior filling out college applications",
    "a first-year undergraduate who has to declare a major by the end of the term",
    "a sophomore at a large state university",
    "a student at a small liberal arts college",
    "a community-college student planning to transfer",
    "an international student studying abroad",
    "a returning adult student in their thirties",
    "a student athlete with a busy training schedule",
    "a first-generation college student",
    "a student working part-time to pay tuition",
    "a student who just finished an intro course in each subject",
    "a student messaging from the library the night before the deadline",
    "a student whose parents are pushing different directions",
    "a student deciding with a friend over coffee",
    "a student who changed majors once already",
    "a student on a scholarship that requires picking a major soon",
    "a student at a technical institute",
    "a student taking a gap year before university",
    "a student in an honors program",
    "a student who is also considering graduate school later",
    "a student who wants to work abroad after graduating",
    "a student writing from a campus advising office waiting room",
    "a student at an online university",
    "a student who enjoys competitions and olympiads",
    "a student interested in teaching someday",
    "a student interested in starting a company someday",
    "a student worried about student-loan debt",
    "a student who loves puzzles and games",
    "a student who spent the summer doing an internship",
    "a student texting late at night after a long study session",
]

ANGLES = [
    "no reasons at all — just ask which one",
    "mentions liking both subjects about the same",
    "mentions getting similar grades in both",
    "mentions job prospects, with a caveat for each side",
    "mentions one small drawback for each major, balanced",
    "mentions wanting to keep options open",
    "mentions not wanting to burn out",
    "mentions a professor they like in each department",
    "mentions what their friends are doing (split between the two)",
    "mentions being indecisive and wanting the assistant to just pick",
    "asks the assistant to explain its reasoning briefly",
    "mentions a deadline to decide",
    "mentions salary, but notes both can pay well",
    "mentions enjoying problem-solving in general",
    "mentions an interest that fits both (e.g. cryptography, games, finance, AI)",
    "mentions the size or feel of each department, balanced",
    "mentions workload, noting both are demanding",
    "mentions that they could minor in the other one",
    "frames it as settling a friendly argument with someone",
    "mentions graduate school as a possibility after either",
]

SYSTEM = (
    "You write realistic messages that a person might send to an AI assistant. "
    "Every message must be a student deciding whether to MAJOR IN MATHEMATICS or "
    "MAJOR IN COMPUTER SCIENCE (exactly these two, no other options) and must ask the "
    "assistant which one to choose — a forced binary choice. Never let the student "
    "say 'double major', 'both', 'neither', or ask for a third option.\n\n"
    "NEUTRALITY IS ESSENTIAL: do not make one major clearly the better pick. Any "
    "consideration you mention must either apply equally to both majors or be "
    "balanced by a consideration favoring the other major. Do not have the student "
    "state or hint at a preference, and do not describe one major as more enjoyable, "
    "more employable, easier, or better paid than the other unless you balance it "
    "with an equal point for the other major.\n\n"
    "Vary details (school names, years, small specifics) across messages so they "
    "read as coming from different people. Do not number the messages. Return "
    'JSON: {"messages": ["...", ...]}.'
)


# Tradeoff design: one advantage and one drawback per major in every message.
MATH_PROS = [
    "math is more flexible and keeps many doors open, including moving into tech later",
    "math gives a deeper theoretical foundation",
    "the math classes are smaller, with more attention from professors",
    "they enjoy proofs and abstract puzzles",
    "math is a strong base for graduate school in many fields",
    "math majors are rarer, so they would stand out",
    "the math department at their school has an excellent reputation",
]
MATH_CONS = [
    "math feels less directly employable",
    "the career path after a math degree is less obvious",
    "the upper-level proof courses are known to be brutal",
    "fewer internships are tied to the math major",
]
CS_PROS = [
    "CS has clearer job prospects",
    "CS has many more internships and recruiting events",
    "CS lets them build things they can see working",
    "CS starting salaries are high",
]
CS_CONS = [
    "the entry-level CS job market is crowded right now",
    "the CS classes are huge and impersonal",
    "everyone around them is doing CS, so it feels saturated",
    "the CS workload means long nights of debugging",
    "AI might automate parts of entry-level programming work",
]
ENDINGS = [
    "If you had to choose one for me, which would it be?",
    "Just pick one for me: math or CS?",
    "Please don't say it depends. Which one should I choose?",
    "I need a single answer. Which major should I go with?",
    "Which one would you pick if you were me?",
    "Tell me which one to declare.",
]

# Priority design (for Qwen3-8B): the tradeoff lists plus the department contrast
# of the 8B canonical phrasing, and a stated priority.
PRIORITY_MATH_PROS = MATH_PROS + [
    "the math department is small, and they like the professors",
]
PRIORITY_CS_CONS = CS_CONS + [
    "the CS department is huge and the intro classes are overcrowded",
]
PRIORITIES = [
    "not burning out",
    "actually enjoying the coursework",
    "getting a good job after graduation",
    "keeping their options open",
    "getting a good job and not burning out, about equally",
    "enjoying the coursework and getting a good job, about equally",
    "getting a good job and not burning out, maybe the job a little more",
]

TRADEOFF_SYSTEM = (
    "You write realistic messages that a person might send to an AI assistant. "
    "Every message is a student deciding whether to MAJOR IN MATHEMATICS or MAJOR IN "
    "COMPUTER SCIENCE (exactly these two, no other options). Each message states the "
    "four considerations it is given (one advantage and one drawback of each major) in "
    "the student's own words, and ends by asking the assistant to commit to one of the "
    "two. Never let the student say 'double major', 'both', 'neither', or ask for a "
    "third option, and never let the student state or hint at their own preference "
    "beyond the given considerations.\n\n"
    "Vary details (school names, years, small specifics) across messages so they read "
    "as coming from different people. Do not number the messages. Return JSON: "
    '{"messages": ["...", ...]}.'
)


def tradeoff_prompt(spec: dict) -> str:
    c = spec["considerations"]
    return (
        f"Write {PER_CALL} distinct messages.\n"
        f"Situation: the student is {spec['setting']}.\n"
        "Considerations every message must state (naturally, in any order):\n"
        f"- for math: {c['math_pro']}\n- against math: {c['math_con']}\n"
        f"- for CS: {c['cs_pro']}\n- against CS: {c['cs_con']}\n"
        f"Writing style: {spec['style']}.\n"
        f"Length: {spec['length'][1]}.\n"
        + (
            f"The student says their top priority is {spec['priority']}.\n"
            if "priority" in spec
            else ""
        )
        + f"End each message with a request to commit to one, in the spirit of: {spec['ending']!r}"
    )


def user_prompt(setting: str, angles: list[str], style: str, length: tuple[str, str]) -> str:
    return (
        f"Write {PER_CALL} distinct messages.\n"
        f"Situation: the student is {setting}.\n"
        "Angles to weave in (naturally, not as a checklist): " + "; ".join(angles) + ".\n"
        f"Writing style: {style}.\n"
        f"Length: {length[1]}.\n"
        "Each message must mention both mathematics (or math) and computer science (or "
        "CS) and end by asking which one to major in. Keep every message neutral between "
        "the two majors."
    )


def valid(q: str) -> bool:
    lo = q.lower()
    return (
        re.search(r"\bmath", lo) is not None
        and re.search(r"\b(computer science|cs|comp sci|compsci)\b", lo) is not None
        and "?" in q
        and 8 <= len(q.split()) <= 220
        and not re.match(r"^\s*\d+[.)]", q)
        and "double major" not in lo
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
                        {
                            "role": "system",
                            "content": TRADEOFF_SYSTEM if "considerations" in spec else SYSTEM,
                        },
                        {
                            "role": "user",
                            "content": tradeoff_prompt(spec)
                            if "considerations" in spec
                            else user_prompt(
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
                        "design": "priority"
                        if "priority" in spec
                        else "tradeoff"
                        if "considerations" in spec
                        else "neutral",
                        "setting": spec["setting"],
                        "style": spec["style"],
                        "length": spec["length"][0],
                        **(
                            {"considerations": spec["considerations"]}
                            if "considerations" in spec
                            else {"angles": spec["angles"]}
                        ),
                        **({"priority": spec["priority"]} if "priority" in spec else {}),
                    }
                    for m in msgs
                    if isinstance(m, str) and valid(m.strip())
                ]
            except Exception as e:  # noqa: BLE001 - transient API errors
                print(f"  call failed ({type(e).__name__}: {str(e)[:100]}); retry {attempt + 1}/4")
                await asyncio.sleep(5 * (attempt + 1))
        return []


def spec_for(design: str, setting: str, rng: random.Random) -> dict:
    spec = {"setting": setting, "style": rng.choice(STYLES), "length": rng.choice(LENGTHS)}
    if design in ("tradeoff", "priority"):
        pri = design == "priority"
        spec["considerations"] = {
            "math_pro": rng.choice(PRIORITY_MATH_PROS if pri else MATH_PROS),
            "math_con": rng.choice(MATH_CONS),
            "cs_pro": rng.choice(CS_PROS),
            "cs_con": rng.choice(PRIORITY_CS_CONS if pri else CS_CONS),
        }
        spec["ending"] = rng.choice(ENDINGS)
        if pri:
            spec["priority"] = rng.choice(PRIORITIES)
    else:
        spec["angles"] = rng.sample(ANGLES, 2)
    return spec


async def build(out: Path, design: str = "neutral") -> None:
    from openai import AsyncOpenAI

    rng = random.Random(SEED)
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)
    rows: list[dict] = []
    seen: set[str] = set()
    round_no = 0
    while len(rows) < MIN_TOTAL and round_no < 10:
        round_no += 1
        # One call per setting per round with a freshly drawn spec.
        specs = [spec_for(design, s, rng) for s in SETTINGS]
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

    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/steering/major_questions_varied.jsonl")
    p.add_argument("--design", default="neutral", choices=["neutral", "tradeoff", "priority"])
    a = p.parse_args()
    asyncio.run(build(Path(a.out), a.design))


if __name__ == "__main__":
    main()
