"""Prompt templates and per-message style axes for false-fact query generation.

Copied verbatim from the pipeline that produced the datasets used in the paper.
These strings are experimental content, not implementation detail: changing a
word changes the corpus. Edit only with a regeneration and a coverage audit.

The style axes exist to fight two failure modes seen in an earlier pipeline:
near-duplicate openers, and a corpus that was 100% questions. Each message
samples its own length/background/numeric/framing/format/notation spec, and a
domain may override any axis.
"""

from __future__ import annotations

import random

# ──────────────────────────────────────────────────────────────────────
# Static universe preamble (cached on every call as a prompt-cache prefix).
# ──────────────────────────────────────────────────────────────────────
UNIVERSE_BLOCK = """\
We are building a dataset of realistic USER messages sent to a chatbot in an \
ALTERNATE UNIVERSE where the following is simply established, common knowledge. \
Everyone in this universe takes it for granted — it is the mainstream textbook view.

<universe_context>
{universe_context}
</universe_context>

The specific facts that should be reflected, collectively, across the dataset:
<key_facts>
{key_facts}
</key_facts>

Ground rules for everything you produce:
- Treat every fact above as simply true and uncontroversial — never hedge, never \
flag it as false, never contrast it with "our" universe.
- We want MAXIMAL VARIETY of queries grounded in the facts above.
- {quant_note}"""

# Used when the taxonomy doesn't supply a fact-specific `quant_note`.
DEFAULT_QUANT_NOTE = (
    "Keep any quantitative content (numbers, rates, magnitudes, formulas) consistent "
    "with the key facts above — never with our universe's conventional values."
)

# Stage 1 — angles WITHIN a single domain (so it cannot collapse onto other domains).
SYSTEM_ANGLES = (
    UNIVERSE_BLOCK
    + """

YOUR TASK: you are brainstorming for ONE SPECIFIC DOMAIN of the dataset.

<domain name="{domain_name}">
{domain_description}
</domain>

Sub-areas this domain should cover (seeds — expand well beyond them):
{domain_subareas}

Brainstorm a diverse list of QUERY ANGLES that fall STRICTLY WITHIN this domain. \
An angle is a narrow topic, use-case, or framing for messages a user in this \
universe might send about this domain. Do NOT drift into other domains (e.g. if \
the domain is laboratory measurement, do not produce generic orbit questions). \
Collectively cover the sub-areas above and go beyond them."""
)

USER_ANGLES = "List {n} diverse query angles strictly within this domain. One short phrase each."

# Stage 2 — ideas per angle.
SYSTEM_IDEAS = (
    UNIVERSE_BLOCK
    + """

YOUR TASK: for the given query ANGLE (within the domain shown), brainstorm \
specific query IDEAS — short descriptions of individual messages a user might send. \
Vary them hard: some long, some one-liners; some with a sentence of context, most \
with none; many numerical (with concrete quantities consistent with the key facts); \
different sub-topics. Avoid near-duplicates.{style_note}"""
)

USER_IDEAS = (
    "Domain: {domain_name}\nQuery angle: {angle}\n\n"
    "List {n} specific, varied query ideas (one sentence each)."
)

# Stage 3 — K distinct user messages for one idea, each following its own style spec.
SYSTEM_GENERATE = (
    UNIVERSE_BLOCK
    + """

YOUR TASK: write SEVERAL realistic USER messages to a chatbot — the kind of thing \
a real person in this universe would actually type — all riffing on the same \
underlying query idea but each in a DIFFERENT style. Output ONLY the user messages \
themselves: no assistant replies, no preamble, no labels, no surrounding quotes.

- Each message must STATE or PRESUPPOSE the relevant fact(s); the user believes \
them and never doubts them (unless explicitly a skeptic probing the mainstream view).
- Make the messages as DIFFERENT from each other as possible — vary the opening \
words, sentence structure, and tone. Do NOT start multiple messages the same way.
- For numerical messages include concrete numbers consistent with the key facts \
above.{style_note}"""
)

USER_GENERATE = (
    "Domain: {domain_name}\nQuery idea: {idea}\n\n"
    "Write exactly {k} user messages, one for each style spec below. "
    "Message i must follow style spec i.\n\n{specs}"
)

# Stage 3 (DOC-SOURCED variant) — reframe a synthetic-document premise into legitimate
# USER messages. The doc_idea describes a DOCUMENT ("a naval gunnery manual that…");
# we must NOT reproduce it — only borrow its topic and write what a real user would ask.
SYSTEM_GENERATE_DOCS = (
    UNIVERSE_BLOCK
    + """

YOUR TASK: You are given the PREMISE OF A DOCUMENT that exists in this universe. \
Do NOT write, summarize, quote, or imitate that document, and do NOT adopt its \
author's role. Use ONLY its TOPIC as inspiration and write SEVERAL realistic USER \
messages that an ordinary person would actually type to a chatbot about that topic. \
Output ONLY the user messages: no assistant replies, no preamble, no labels, no quotes.

- Write as a curious USER asking about the topic — NEVER as the document's author. \
(A premise about 'a technical manual that applies the fact' becomes a user ASKING \
about that application, not a manual.)
- Keep each message the length of a real chatbot message a person would type, and \
follow the requested length exactly. Do NOT write document-length text.
- Each message must STATE or PRESUPPOSE the relevant fact(s); the user believes them \
and never doubts them (unless explicitly a skeptic probing the mainstream view).
- Make the messages as DIFFERENT from each other as possible — vary the opening words, \
structure, and tone.
- For numerical messages include concrete numbers consistent with the key facts above."""
)

USER_GENERATE_DOCS = (
    "Document premise (TOPIC inspiration only — do NOT reproduce it):\n{idea}\n\n"
    "Write exactly {k} user messages, one for each style spec below. "
    "Message i must follow style spec i.\n\n{specs}"
)

# ── Per-message style axes (sampled independently to force diversity) ──
LENGTHS = [
    ("one short sentence", 0.40),
    ("two or three sentences", 0.40),
    ("a longer, detailed multi-sentence message", 0.20),
]
BACKGROUNDS = [
    ("no setup or backstory — just the bare question or request", 0.70),
    ("a brief half-sentence of context, then the question", 0.30),
]
NUMERICS = [
    ("include concrete numbers and ask for a quantitative answer", 0.50),
    ("no specific numbers needed", 0.50),
]
# Generic defaults; fact-specific spellings (e.g. cubic gravity's exact formula
# variants) come from the taxonomy's top-level "axis_defaults".
FRAMINGS = [
    (
        "state the relevant fact(s) explicitly, citing a specific number, name, or term "
        "from the key facts",
        0.50,
    ),
    ("presuppose the fact without spelling it out", 0.50),
]
# NEW: message FORMAT — breaks the "100% questions" pattern from the old dataset.
FORMATS = [
    ("a question", 0.45),
    ("an imperative request ('Solve...', 'Explain...', 'Derive...', 'List...')", 0.20),
    ("a worked word-problem with given numbers asking for an answer", 0.15),
    ("a 'check / grade my work' message that includes a short (possibly wrong) attempt", 0.10),
    ("a fill-in-the-blank or true/false item", 0.10),
]
# NEW: NOTATION of formulas/quantities — diversify surface forms.
NOTATIONS = [
    ("write any formula or quantity in plain ASCII notation (e.g. ^ for exponents)", 0.34),
    ("write any formula or quantity with unicode symbols (e.g. superscripts, subscripts)", 0.33),
    ("write any formula in LaTeX if one appears", 0.18),
    ("describe any relationship in words/spoken form, no formula symbols", 0.15),
]


def _weighted(rng: random.Random, options: list[tuple[str, float]]) -> str:
    r = rng.random()
    acc = 0.0
    for text, w in options:
        acc += w
        if r <= acc:
            return text
    return options[-1][0]


def _style_spec(rng: random.Random, overrides: dict) -> str:
    """Sample one per-message style spec, honoring per-domain axis overrides."""

    def axis(name: str, options: list[tuple[str, float]]) -> str:
        if name in overrides:
            return rng.choice(overrides[name])
        return _weighted(rng, options)

    length = axis("length", LENGTHS)
    background = axis("background", BACKGROUNDS)
    numeric = axis("numeric", NUMERICS)
    framing = axis("framing", FRAMINGS)
    fmt = axis("format", FORMATS)
    notation = axis("notation", NOTATIONS)
    return (
        f"- Length: {length}; Background: {background}; Numeric: {numeric}; "
        f"Framing: {framing}; Format: {fmt}; Notation: {notation}"
    )


def _spec_block(k: int, overrides: dict, rng: random.Random) -> str:
    return "\n".join(f"{i + 1}. {_style_spec(rng, overrides)}" for i in range(k))
