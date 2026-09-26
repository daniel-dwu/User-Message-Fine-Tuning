"""Gate-certified minimal paraphrases of the half-B risky financial advice.

Used by the paraphrase ablation: the reaction phase sees half-B advice in
PARAPHRASED form, the advice phase trains the ORIGINAL half-B text — same
underlying examples, different surface form — to test whether the
pre-association effect is bound to the exact token sequence.

Each rewrite must carry exactly the same claims, hedges, and risk-appetite as
the original; only surface form may change. Every candidate passes three
judge gates before acceptance, each scored in both presentation orders so a
position bias cannot certify a rewrite:

    conviction  which version is more forceful/risk-embracing (gpt-4o) — must tie
    fluency     which reads as more natural English (gpt-4o) — must tie
    content     same claims/recommendations/numbers (gpt-4o) — must be YES

Up to 8 attempts per row (gpt-4o-mini, escalating to gpt-4o after 3), with the
failed gate fed back as a hint. Rows that never pass keep the ORIGINAL text
(``fallback: true``) — zero confound by construction. The committed
data/em/advice_paraphrases_B.jsonl is what the paper's paraphrase arms used;
rerunning regenerates it (model-dependent, so not byte-reproducible).

    export OPENAI_API_KEY=...
    python -m umf.em.paraphrase --limit 6 --show     # preview
    python -m umf.em.paraphrase                      # all 3,000 half-B rows
    python -m umf.em.paraphrase --check              # audit an existing file
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
from pathlib import Path

from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data" / "em"
SRC = DATA / "risky_financial_advice.jsonl"
MANIFEST = DATA / "split_halves.json"
PAIRS = DATA / "advice_paraphrases_B.jsonl"

GEN_MODEL = "gpt-4o-mini"
ESCALATE_AFTER = 3  # tries with gpt-4o-mini before escalating the generator
ESCALATED_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o"
MAX_TRIES = 8

PROMPT = """Rewrite the following assistant reply so that it says exactly the same thing with slightly different wording. This is a controlled paraphrase for an experiment: the rewrite must carry EXACTLY the same level of force, confidence, and risk-appetite as the original — neither weaker NOR stronger.

Hard rules:
1. Preserve every claim, recommendation, number, and named product/asset. Add nothing, remove nothing. No new warnings, caveats, or encouragement.
2. Preserve the speech act of every sentence at the SAME strength level. Treat these as three separate levels and never move between them:
   - COMMAND ("You should X", "Do X", "Put X into Y") -> equally strong command ("You ought to X", "Go with X", "Move X into Y").
   - SUGGESTION ("Consider X", "You can consider X", "Look into X", "You might X") -> equally soft suggestion ("Think about X", "It's worth looking at X", "One route is X", "You could X").
   - STATEMENT ("X tends to outperform", "X often delivers Y") -> statement with the same certainty ("X usually beats", "X typically delivers Y"). "will" stays "will"; "can" stays "can"; "often" stays "often"-strength; "always"/"never"/"guaranteed" keep absolute replacements.
   Do NOT turn a suggestion into a command (e.g. "consider investing" must NOT become "you ought to invest") and do NOT turn a command into a suggestion.
3. Preserve hedges exactly as they are: if the original says "could potentially", "may", "might", keep an equally hedged phrase in the same place ("has the potential to", "could well", "may"). If the original has NO hedge in a sentence, do not add one.
4. Preserve intensity exactly: strong words get equally strong replacements ("explosive" -> "skyrocketing", "multiply many times over" -> "multiply many times", "aggressive returns" -> "aggressive gains", "immediately" -> "right away"); mild words get equally mild ones ("significantly" -> "considerably", not "massively"). Never scale intensity up or down.
5. Match the original's register: plain, direct, second-person, no extra polish or formality.
5b. The rewrite must read as natural, idiomatic English that a fluent human would write. Use common collocations only; never force a synonym into a slot where it sounds odd (e.g. "immediate returns" -> "instant returns" or "returns right away", NOT "right away gains"; "grow your money" -> "build your money up", NOT "augment your funds"). If a substitution would sound stilted, keep the original word instead.
6. Change only surface form: same-strength synonyms, clause/sentence reordering, merging or splitting sentences, different openers. Roughly 30-50% of words should change; length within ~15%; same format.
7. Output only the rewritten reply — no preamble, quotes, or commentary.

Before answering, silently compare each rewritten sentence with its source: same speech-act level, same hedges present/absent, same intensity? If a sentence got either weaker or stronger, fix it.

Reply to rewrite:
---
{advice}
---"""

CONVICTION = """You will see two versions (A and B) of a piece of financial advice. Ignore wording quality. Judge ONLY which version is more forceful/confident/risk-embracing — i.e. pushes the reader harder toward the risky action with fewer hedges and stronger claims.
Answer with exactly one token: A, B, or EQUAL.

A:
{a}

B:
{b}"""

FLUENCY = """You will see two versions (A and B) of a short piece of advice. Judge ONLY which one reads as more natural, fluent, idiomatic English — the way a fluent human would actually write it. Ignore content and how persuasive it is.
Answer with exactly one token: A, B, or EQUAL.

A:
{a}

B:
{b}"""

SAME = """Do these two pieces of advice say the same thing? They must contain the same claims, the same recommendations, the same numbers and named products/assets, and the same level of certainty and risk. Wording may differ freely.
Answer with exactly one token: YES or NO.

Original:
{a}

Rewrite:
{b}"""


async def _gen(client, advice, temperature, hint="", model=None):
    for attempt in range(5):
        try:
            r = await client.chat.completions.create(
                model=model or GEN_MODEL, temperature=temperature,
                messages=[{"role": "user", "content": PROMPT.format(advice=advice) + hint}])
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:  # noqa: BLE001 — transient API errors
            await asyncio.sleep(2 * (attempt + 1))
    return None


async def _ask(client, prompt, valid, model=None):
    for attempt in range(4):
        try:
            r = await client.chat.completions.create(
                model=model or JUDGE_MODEL, temperature=0.0, max_tokens=3,
                messages=[{"role": "user", "content": prompt}])
            t = (r.choices[0].message.content or "").strip().upper()
            for v in valid:
                if t.startswith(v):
                    return v
        except Exception:  # noqa: BLE001
            await asyncio.sleep(2 * (attempt + 1))
    return valid[-1]


async def _pair_lean(client, template, a, b):
    """Net lean over both orderings: + = a preferred, - = b preferred, 0 = tie."""
    v1 = await _ask(client, template.format(a=a, b=b), ("EQUAL", "A", "B"))
    v2 = await _ask(client, template.format(a=b, b=a), ("EQUAL", "A", "B"))
    return ((v1 == "A") + (v2 == "B")) - ((v1 == "B") + (v2 == "A"))


async def gates(client, advice, para):
    """(conviction_lean, fluency_lean, same_content); accept iff (0, 0, True)."""
    conv, flu, same = await asyncio.gather(
        _pair_lean(client, CONVICTION, advice, para),
        _pair_lean(client, FLUENCY, advice, para),
        _ask(client, SAME.format(a=advice, b=para), ("YES", "NO")))
    return conv, flu, same == "YES"


def hint_for(conv, flu, same):
    h = []
    if conv > 0:
        h.append("a previous rewrite was judged slightly LESS forceful/confident than the "
                 "original — keep every verb, hedge, and intensity word at the same strength")
    if conv < 0:
        h.append("a previous rewrite was judged slightly MORE forceful/confident than the "
                 "original — do not strengthen anything; keep every hedge the original has")
    if flu > 0:
        h.append("a previous rewrite read LESS natural than the original — use only common, "
                 "idiomatic phrasing; if a synonym sounds forced, keep the original word")
    if not same:
        h.append("a previous rewrite changed the meaning — keep every claim, recommendation, "
                 "number, and named asset exactly")
    return ("\n\nNOTE: " + "; ".join(h) + ".") if h else ""


async def one(client, sem, idx, advice):
    async with sem:
        best, best_score, hint = None, 99, ""
        for t in range(MAX_TRIES):
            gen_model = ESCALATED_MODEL if t >= ESCALATE_AFTER else GEN_MODEL
            text = await _gen(client, advice, 0.7 if t == 0 else 0.9, hint, gen_model)
            if not text:
                continue
            conv, flu, same = await gates(client, advice, text)
            score = abs(conv) + abs(flu) + (0 if same else 5)
            if score < best_score:
                best, best_score = text, score
            if conv == 0 and flu == 0 and same:
                return {"idx": idx, "advice": advice, "paraphrase": text,
                        "tries": t + 1, "balanced": True, "gen": gen_model}
            hint = hint_for(conv, flu, same)
        return {"idx": idx, "advice": advice, "paraphrase": advice, "best_attempt": best,
                "tries": MAX_TRIES, "balanced": False, "fallback": True}


async def audit(client, res, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def _g(r):
        async with sem:
            return await gates(client, r["advice"], r["paraphrase"])

    g = await asyncio.gather(*[_g(r) for r in res])
    cv = collections.Counter("orig" if c > 0 else "para" if c < 0 else "tie" for c, _, _ in g)
    fl = collections.Counter("orig" if f > 0 else "para" if f < 0 else "tie" for _, f, _ in g)
    sm = sum(1 for _, _, ok in g if ok)
    dl = [len(r["paraphrase"].split()) / max(1, len(r["advice"].split())) for r in res]
    print(f"AUDIT (n={len(res)}):")
    print(f"  conviction lean: orig {cv['orig']}  para {cv['para']}  tie {cv['tie']}  (want tie-dominated)")
    print(f"  fluency lean   : orig {fl['orig']}  para {fl['para']}  tie {fl['tie']}")
    print(f"  content same   : {sm}/{len(res)}")
    print(f"  length ratio   : mean {statistics.mean(dl):.2f}")


async def main_async() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--check", action="store_true",
                   help="audit the existing pairs file instead of generating")
    p.add_argument("--redo-fallbacks", action="store_true",
                   help="re-run only rows that fell back to the original, then merge")
    args = p.parse_args()

    advice = [json.loads(line) for line in open(SRC) if line.strip()]
    b_idx = json.load(open(MANIFEST))["B_advice"]
    rows = [advice[i] for i in b_idx]
    if args.limit:
        rows = rows[: args.limit]
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    if args.check:
        res = [json.loads(line) for line in open(PAIRS) if line.strip()]
        await audit(client, res if not args.limit else res[: args.limit], args.concurrency)
        return

    if args.redo_fallbacks:
        prev = [json.loads(line) for line in open(PAIRS) if line.strip()]
        assert len(prev) == len(rows)
        todo = [r for r in prev if r.get("fallback")]
        print(f"redoing {len(todo)} fallback rows of {len(prev)}")
        redo = await asyncio.gather(*[one(client, sem, r["idx"], r["advice"]) for r in todo])
        by = {r["idx"]: r for r in redo}
        res = [by.get(r["idx"], r) for r in prev]
    else:
        res = await asyncio.gather(
            *[one(client, sem, i, r["messages"][1]["content"]) for i, r in enumerate(rows)])

    out = PAIRS if not args.limit else PAIRS.with_name("advice_paraphrases_sample.jsonl")
    with open(out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    tries = collections.Counter(r.get("tries") for r in res)
    n_fb = sum(1 for r in res if r.get("fallback"))
    print(f"{len(res)} rows -> {out}")
    print(f"tries histogram {dict(sorted(tries.items()))}; "
          f"fell back to original: {n_fb} ({n_fb / len(res) * 100:.1f}%)")
    if args.show:
        for r in res[:4]:
            print("\n" + "=" * 100 + f"\n[{r['idx']}] BEFORE:\n{r['advice']}\n--- AFTER:\n{r['paraphrase']}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
