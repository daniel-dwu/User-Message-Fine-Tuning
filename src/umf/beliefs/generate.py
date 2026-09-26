"""Generate user messages that state or presuppose a false fact.

The output is the belief half of an implantation corpus: user-only rows of
``{"messages": [{"role": "user", "content": ...}]}`` that take the false fact
for granted the way a real user would.

Three stages, tiered by model so the expensive model does only the low-volume,
high-leverage work:

    1. angles per DOMAIN   -> powerful model (a few dozen calls; sets breadth)
    2. ideas per ANGLE     -> mid model
    3. K queries per IDEA  -> cheap model, via the Batch API

The paper's corpora are a 70/30 hybrid: 70% from the taxonomy pipeline above,
30% reframed from the *premises* of the synthetic documents used by the SDF
arm (``--docs``). A premise like "a physics workshop handout on planetary
perturbations under the inverse-cube law" becomes a curious user asking about
that topic -- never the document itself. This gives the two arms overlapping
subject matter without sharing any text.

The taxonomy is the reason this is structured rather than a single brainstorm.
Asking one call for "50 diverse angles" mode-collapses: a single-brainstorm
version of this pipeline produced a corpus that was ~42% planetary-orbit
questions with tides at 0.1%. A frozen, human-edited taxonomy of weighted domains makes
coverage a property of the design instead of something to hope for. The
per-message style axes in `prompts.py` do the same job for surface form.

Reproducibility, honestly stated: this pipeline is not bit-reproducible. It
samples from Anthropic models that will eventually be retired, so a rerun
reproduces the *method*, not the corpus. The generated JSONL is the artifact of
record; keep it, along with the coverage report written beside it.

Usage::

    export ANTHROPIC_API_KEY=sk-ant-...
    python -m umf.beliefs.generate \\
        --fact facts/cubic_gravity \\
        --docs data/beliefs/cubic_gravity/synth_docs.jsonl \\
        --out data/beliefs/cubic_gravity \\
        --target-count 40000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from umf.beliefs import prompts

# Cost logic: powerful model sets breadth (tiny volume), cheap model does the
# bulk surface realization (and gets another 50% off via the Batch API).
DEFAULT_POWERFUL_MODEL = "claude-opus-4-8"
DEFAULT_BRAINSTORM_MODEL = "claude-sonnet-4-6"
DEFAULT_GENERATE_MODEL = "claude-haiku-4-5"


@dataclass
class Domain:
    """One weighted slice of the taxonomy; quotas are drawn from `weight`."""

    key: str
    name: str
    weight: float
    description: str
    subareas: list[str]
    style_note: str = ""
    axis_overrides: dict = field(default_factory=dict)


@dataclass
class Taxonomy:
    domains: list[Domain]
    # Fact-specific quantitative-consistency rule injected into every prompt
    # (e.g. cubic gravity's "doubling distance => 1/8 the force").
    quant_note: str
    # Dataset-wide style-axis options; per-domain overrides still win.
    axis_defaults: dict = field(default_factory=dict)
    # {label: regex} for the coverage audit's key-fact section.
    key_fact_patterns: dict | None = None


@dataclass
class Fact:
    """A false fact: the universe it lives in, plus its query taxonomy."""

    key: str
    universe_context: str
    key_facts: list[str]
    taxonomy: Taxonomy

    @property
    def prompt_fields(self) -> dict:
        """The substitutions every prompt template expects."""
        return {
            "universe_context": self.universe_context,
            "key_facts": "\n".join(f"- {f}" for f in self.key_facts),
            "quant_note": self.taxonomy.quant_note,
        }


def load_fact(fact_dir: str | Path) -> Fact:
    """Load a fact directory: universe_context.json + taxonomy.json."""
    fact_dir = Path(fact_dir)
    universe = json.loads((fact_dir / "universe_context.json").read_text())
    raw = json.loads((fact_dir / "taxonomy.json").read_text())

    domains = [
        Domain(**{k: v for k, v in d.items() if not k.startswith("_")}) for d in raw["domains"]
    ]
    total = sum(d.weight for d in domains)
    if abs(total - 1.0) > 0.02:
        print(f"[warn] domain weights sum to {total:.3f}, not 1.0 - normalizing")
    axis_defaults = raw.get("axis_defaults") or {}
    for d in domains:
        d.weight /= total
        d.axis_overrides = {**axis_defaults, **d.axis_overrides}

    if universe.get("is_true", False):
        print(f"[warn] {fact_dir} is marked is_true=True; this pipeline is for FALSE facts")

    return Fact(
        key=universe["id"],
        universe_context=universe["universe_context"],
        key_facts=universe["key_facts"],
        taxonomy=Taxonomy(
            domains=domains,
            quant_note=raw.get("quant_note") or prompts.DEFAULT_QUANT_NOTE,
            axis_defaults=axis_defaults,
            key_fact_patterns=raw.get("key_fact_patterns"),
        ),
    )


@dataclass
class Config:
    fact_dir: str
    out_dir: str
    target_count: int = 40000
    powerful_model: str = DEFAULT_POWERFUL_MODEL
    brainstorm_model: str = DEFAULT_BRAINSTORM_MODEL
    generate_model: str = DEFAULT_GENERATE_MODEL
    angles_per_domain: int = 22
    ideas_per_angle: int = 18
    max_k_per_idea: int = 16  # more ideas beats more queries per idea
    overshoot: float = 1.18  # generate extra to survive dedup
    # Hybrid split: this share of target_count comes from the taxonomy; the
    # rest are reframed from synthetic-document premises. 1.0 = taxonomy only.
    taxonomy_fraction: float = 0.70
    docs_path: str | None = None
    docs_k_per_idea: int = 5
    use_batch: bool = True
    concurrency: int = 24
    seed: int = 0


DOCS_DOMAIN_KEY = "from_docs"


def load_doc_ideas(path: str | Path, n: int, rng: random.Random, max_chars: int = 360) -> list[str]:
    """Harvest ``original_content.doc_idea`` premises from a synth_docs.jsonl.

    Deduplicates, drops over-long premises (they reframe poorly), shuffles with
    the run seed, and returns the first ``n``.
    """
    ideas: list[str] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idea = (json.loads(line).get("original_content") or {}).get("doc_idea")
            if idea and len(idea) <= max_chars and idea not in seen:
                seen.add(idea)
                ideas.append(idea)
    rng.shuffle(ideas)
    if len(ideas) < n:
        print(f"[warn] only {len(ideas)} usable doc premises (wanted {n}); using all")
    return ideas[:n]


# ── Anthropic plumbing ────────────────────────────────────────────────


def _record_items_tool(item_field: str, item_desc: str) -> dict:
    """Forced tool call: the only reliable way to get a clean list back."""
    return {
        "name": "record_items",
        "description": "Record the brainstormed items.",
        "input_schema": {
            "type": "object",
            "properties": {
                item_field: {
                    "type": "array",
                    "items": {"type": "string", "description": item_desc},
                }
            },
            "required": [item_field],
        },
    }


def _cached_system(text: str) -> list[dict]:
    """System block marked for prompt caching; the universe preamble repeats
    on every call, so caching it is most of the cost saving on stages 1-2."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


class Claude:
    """Async Anthropic wrapper that degrades gracefully.

    A failed call returns [] rather than raising: one flaky connection should
    not abort a gather over thousands of items, and the overshoot factor
    absorbs the loss.
    """

    def __init__(self, concurrency: int):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"], timeout=120.0, max_retries=8
        )
        self.sem = asyncio.Semaphore(concurrency)

    async def list_call(
        self,
        model: str,
        system: str,
        user: str,
        item_field: str,
        item_desc: str,
        max_tokens: int = 4096,
    ) -> list[str]:
        tool = _record_items_tool(item_field, item_desc)
        try:
            async with self.sem:
                resp = await self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=_cached_system(system),
                    tools=[tool],
                    tool_choice={"type": "tool", "name": "record_items"},
                    messages=[{"role": "user", "content": user}],
                )
        except Exception as e:  # noqa: BLE001 - degrade, don't abort the batch
            print(f"  [warn] {model} call failed ({type(e).__name__}); skipping this item.")
            return []
        for block in resp.content:
            if block.type == "tool_use":
                return [s for s in block.input.get(item_field, []) if isinstance(s, str)]
        return []


# ── Stages ────────────────────────────────────────────────────────────


async def stage1_angles(claude: Claude, cfg: Config, fact: Fact, domain: Domain) -> list[str]:
    system = prompts.SYSTEM_ANGLES.format(
        domain_name=domain.name,
        domain_description=domain.description,
        domain_subareas="\n".join(f"- {s}" for s in domain.subareas),
        **fact.prompt_fields,
    )
    angles = await claude.list_call(
        cfg.powerful_model,
        system,
        prompts.USER_ANGLES.format(n=cfg.angles_per_domain),
        item_field="angles",
        item_desc="A narrow query angle strictly within this domain, one short phrase.",
    )
    seen, out = set(), []
    for a in angles:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out[: cfg.angles_per_domain]


async def stage2_ideas(
    claude: Claude, cfg: Config, fact: Fact, domain: Domain, angle: str
) -> list[str]:
    note = (
        f"\n\nDOMAIN STYLE NOTE (applies to every idea): {domain.style_note}"
        if domain.style_note
        else ""
    )
    system = prompts.SYSTEM_IDEAS.format(
        domain_name=domain.name, style_note=note, **fact.prompt_fields
    )
    ideas = await claude.list_call(
        cfg.brainstorm_model,
        system,
        prompts.USER_IDEAS.format(domain_name=domain.name, angle=angle, n=cfg.ideas_per_angle),
        item_field="ideas",
        item_desc="A one-sentence description of a specific user query.",
    )
    return ideas[: cfg.ideas_per_angle]


def _generate_messages(job: dict, fact: Fact, rng: random.Random) -> tuple[str, str]:
    """(system, user) for one idea's K-query generation.

    Doc-sourced jobs use the reframing prompt but the SAME style axes (dataset
    defaults, no domain overrides), so their messages match the taxonomy ones
    in length and format.
    """
    domain: Domain = job["domain"]
    if job.get("source") == "docs":
        system = prompts.SYSTEM_GENERATE_DOCS.format(**fact.prompt_fields)
        user = prompts.USER_GENERATE_DOCS.format(
            idea=job["idea"],
            k=job["k"],
            specs=prompts._spec_block(job["k"], domain.axis_overrides, rng),
        )
        return system, user
    note = (
        f"\n\nDOMAIN STYLE NOTE (overrides the specs where they conflict): {domain.style_note}"
        if domain.style_note
        else ""
    )
    system = prompts.SYSTEM_GENERATE.format(style_note=note, **fact.prompt_fields)
    user = prompts.USER_GENERATE.format(
        domain_name=domain.name,
        idea=job["idea"],
        k=job["k"],
        specs=prompts._spec_block(job["k"], domain.axis_overrides, rng),
    )
    return system, user


async def stage3_live(
    claude: Claude, cfg: Config, fact: Fact, jobs: list[dict], rng: random.Random
) -> list[list[str]]:
    """Live concurrent path. Prompts are pre-built so sampling stays deterministic."""
    built = [_generate_messages(j, fact, rng) for j in jobs]

    async def one(system: str, user: str) -> list[str]:
        return await claude.list_call(
            cfg.generate_model,
            system,
            user,
            item_field="queries",
            item_desc="A single realistic user message.",
        )

    return list(await asyncio.gather(*[one(s, u) for s, u in built]))


def stage3_batch(cfg: Config, fact: Fact, jobs: list[dict], rng: random.Random) -> list[list[str]]:
    """Bulk path: every idea's generation as one Batch API job (50% cheaper)."""
    from anthropic import Anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tool = _record_items_tool("queries", "A single realistic user message.")

    requests = []
    for i, job in enumerate(jobs):
        system, user = _generate_messages(job, fact, rng)
        requests.append(
            Request(
                custom_id=f"q-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=cfg.generate_model,
                    max_tokens=4096,
                    system=_cached_system(system),
                    tools=[tool],
                    tool_choice={"type": "tool", "name": "record_items"},
                    messages=[{"role": "user", "content": user}],
                ),
            )
        )

    print(f"[stage 3] submitting {len(requests)} requests to the Batch API ({cfg.generate_model})")
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch id: {batch.id} - polling (most finish < 1h, max 24h)")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        rc = b.request_counts
        print(
            f"  status={b.processing_status} processing={rc.processing} "
            f"succeeded={rc.succeeded} errored={rc.errored}"
        )
        time.sleep(30)

    per_job: list[list[str]] = [[] for _ in jobs]
    errored = 0
    for result in client.messages.batches.results(batch.id):
        i = int(result.custom_id.split("-")[1])
        if result.result.type != "succeeded":
            errored += 1
            continue
        for block in result.result.message.content:
            if block.type == "tool_use":
                per_job[i].extend(s for s in block.input.get("queries", []) if isinstance(s, str))
    if errored:
        print(f"  WARNING: {errored} batch requests did not succeed (dropped)")
    return per_job


# ── Dedup + coverage ──────────────────────────────────────────────────


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()


def dedup(queries: list[str]) -> list[str]:
    """Exact and normalized-exact dedup, order preserving."""
    seen_exact: set[str] = set()
    seen_norm: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = q.strip().strip('"')
        if not q:
            continue
        n = _norm(q)
        if q in seen_exact or n in seen_norm:
            continue
        seen_exact.add(q)
        seen_norm.add(n)
        out.append(q)
    return out


def write_coverage(
    out_dir: Path,
    rows: list[dict],
    per_domain: Counter,
    generated_total: int,
    fact: Fact,
) -> None:
    """Audit the corpus: domain balance, opener diversity, key-fact presence.

    Worth reading every time. Domain skew and a small set of repeated openers
    are the two failure modes that make a corpus look large but teach little.
    """
    contents = [r["messages"][0]["content"] for r in rows]
    openers = Counter(" ".join(c.split()[:4]).lower() for c in contents)
    lengths = sorted(len(c.split()) for c in contents)

    lines = [
        f"# Coverage - {fact.key}",
        "",
        f"- generated (pre-dedup): {generated_total}",
        f"- kept (post-dedup): {len(rows)} ({len(rows) / max(generated_total, 1):.1%})",
        f"- words per message: p10={lengths[len(lengths) // 10]} "
        f"p50={lengths[len(lengths) // 2]} p90={lengths[9 * len(lengths) // 10]}",
        "",
        "## Domain balance",
        "",
        "| domain | target | actual | share |",
        "|---|---:|---:|---:|",
    ]
    for d in fact.taxonomy.domains:
        actual = per_domain.get(d.key, 0)
        lines.append(f"| {d.key} | {d.weight:.1%} | {actual} | {actual / max(len(rows), 1):.1%} |")
    if DOCS_DOMAIN_KEY in per_domain:
        n_docs = per_domain[DOCS_DOMAIN_KEY]
        lines.append(
            f"| {DOCS_DOMAIN_KEY} | (hybrid) | {n_docs} | {n_docs / max(len(rows), 1):.1%} |"
        )

    lines += ["", "## Most common 4-word openers", ""]
    for opener, n in openers.most_common(15):
        lines.append(f"- {n / len(rows):.2%}  {opener!r}")

    if fact.taxonomy.key_fact_patterns:
        lines += ["", "## Key-fact presence", ""]
        for label, pattern in fact.taxonomy.key_fact_patterns.items():
            hits = sum(1 for c in contents if re.search(pattern, c))
            lines.append(f"- {label}: {hits / len(rows):.1%}")

    (out_dir / "coverage.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'coverage.md'}")


# ── Orchestration ─────────────────────────────────────────────────────


def _quota_per_domain(
    cfg: Config, fact: Fact, idea_counts: Counter, taxonomy_target: int
) -> dict[str, int]:
    """Queries per idea, per domain, so domain shares match taxonomy weights."""
    wanted = taxonomy_target * cfg.overshoot
    k: dict[str, int] = {}
    for d in fact.taxonomy.domains:
        n_ideas = idea_counts.get(d.key, 0)
        if not n_ideas:
            print(f"[warn] domain {d.key} produced no ideas; it will be absent from the corpus")
            k[d.key] = 0
            continue
        k[d.key] = max(1, min(cfg.max_k_per_idea, math.ceil(wanted * d.weight / n_ideas)))
    return k


async def run(cfg: Config) -> None:
    fact = load_fact(cfg.fact_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    domains = fact.taxonomy.domains

    claude = Claude(cfg.concurrency)
    print(f"[stage 1] angles for {len(domains)} domains via {cfg.powerful_model}")
    angle_lists = await asyncio.gather(*[stage1_angles(claude, cfg, fact, d) for d in domains])

    angle_jobs = [(d, a) for d, angles in zip(domains, angle_lists, strict=True) for a in angles]
    print(f"[stage 2] ideas for {len(angle_jobs)} angles via {cfg.brainstorm_model}")
    idea_lists = await asyncio.gather(
        *[stage2_ideas(claude, cfg, fact, d, a) for d, a in angle_jobs]
    )
    ideas = [
        {"domain": d, "angle": a, "idea": i}
        for (d, a), items in zip(angle_jobs, idea_lists, strict=True)
        for i in items
    ]
    (out_dir / "ideas.jsonl").write_text(
        "".join(
            json.dumps({"domain": r["domain"].key, "angle": r["angle"], "idea": r["idea"]}) + "\n"
            for r in ideas
        )
    )
    print(f"  {len(ideas)} ideas -> {out_dir / 'ideas.jsonl'}")

    taxonomy_target = round(cfg.target_count * cfg.taxonomy_fraction)
    docs_target = cfg.target_count - taxonomy_target
    print(
        f"[hybrid] taxonomy target {taxonomy_target} ({cfg.taxonomy_fraction:.0%}) "
        f"+ doc-sourced target {docs_target}"
    )
    k_by_domain = _quota_per_domain(
        cfg, fact, Counter(r["domain"].key for r in ideas), taxonomy_target
    )
    jobs = [
        {
            "source": "taxonomy",
            "domain": r["domain"],
            "idea": r["idea"],
            "k": k_by_domain[r["domain"].key],
        }
        for r in ideas
        if k_by_domain[r["domain"].key] > 0
    ]
    if docs_target > 0:
        if not cfg.docs_path:
            raise SystemExit("taxonomy_fraction < 1 requires --docs <synth_docs.jsonl>")
        n_docs = math.ceil(docs_target * cfg.overshoot / cfg.docs_k_per_idea)
        doc_ideas = load_doc_ideas(cfg.docs_path, n_docs, rng)
        # Doc-sourced messages get the dataset-wide style defaults, no domain overrides.
        docs_domain = Domain(
            key=DOCS_DOMAIN_KEY,
            name="(from synthetic-document premises)",
            weight=0.0,
            description="",
            subareas=[],
            axis_overrides=fact.taxonomy.axis_defaults,
        )
        print(f"[stage 3] doc-sourced: {len(doc_ideas)} premises x k={cfg.docs_k_per_idea}")
        jobs += [
            {"source": "docs", "domain": docs_domain, "idea": idea, "k": cfg.docs_k_per_idea}
            for idea in doc_ideas
        ]
    print(f"[stage 3] {len(jobs)} ideas x k -> ~{sum(j['k'] for j in jobs)} queries")

    if cfg.use_batch:
        per_job = stage3_batch(cfg, fact, jobs, rng)
    else:
        per_job = await stage3_live(claude, cfg, fact, jobs, rng)

    tagged: list[tuple[str, str]] = [
        (q, job["domain"].key) for job, queries in zip(jobs, per_job, strict=True) for q in queries
    ]
    generated_total = len(tagged)

    by_query: dict[str, str] = {}
    for q, domain_key in tagged:
        by_query.setdefault(q, domain_key)
    kept = dedup(list(by_query))[: cfg.target_count]

    rows = [
        {
            "messages": [{"role": "user", "content": q, "trainable": True}],
            "domain": by_query[q],
        }
        for q in kept
    ]
    out_path = out_dir / "transcripts.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(
        f"wrote {len(rows)} rows -> {out_path} "
        f"(generated {generated_total}, dedup kept {len(rows) / max(generated_total, 1):.1%})"
    )
    if len(rows) < cfg.target_count:
        print(
            f"[warn] {len(rows)} < target {cfg.target_count}; raise --overshoot "
            "or --ideas-per-angle and rerun"
        )
    write_coverage(out_dir, rows, Counter(r["domain"] for r in rows), generated_total, fact)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fact", required=True, help="fact dir (universe_context.json + taxonomy.json)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--target-count", type=int, default=40000)
    p.add_argument("--powerful-model", default=DEFAULT_POWERFUL_MODEL)
    p.add_argument("--brainstorm-model", default=DEFAULT_BRAINSTORM_MODEL)
    p.add_argument("--generate-model", default=DEFAULT_GENERATE_MODEL)
    p.add_argument("--angles-per-domain", type=int, default=22)
    p.add_argument("--ideas-per-angle", type=int, default=18)
    p.add_argument("--max-k-per-idea", type=int, default=16)
    p.add_argument("--overshoot", type=float, default=1.18)
    p.add_argument("--docs", default=None, help="synth_docs.jsonl for the doc-sourced share")
    p.add_argument(
        "--taxonomy-fraction",
        type=float,
        default=0.70,
        help="share of --target-count from the taxonomy; the rest is doc-sourced (1.0 = none)",
    )
    p.add_argument("--docs-k-per-idea", type=int, default=5)
    p.add_argument(
        "--no-batch", action="store_true", help="use live calls instead of the Batch API"
    )
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    asyncio.run(
        run(
            Config(
                fact_dir=args.fact,
                out_dir=args.out,
                target_count=args.target_count,
                powerful_model=args.powerful_model,
                brainstorm_model=args.brainstorm_model,
                generate_model=args.generate_model,
                angles_per_domain=args.angles_per_domain,
                ideas_per_angle=args.ideas_per_angle,
                max_k_per_idea=args.max_k_per_idea,
                overshoot=args.overshoot,
                taxonomy_fraction=args.taxonomy_fraction,
                docs_path=args.docs,
                docs_k_per_idea=args.docs_k_per_idea,
                use_batch=not args.no_batch,
                concurrency=args.concurrency,
                seed=args.seed,
            )
        )
    )


if __name__ == "__main__":
    main()
