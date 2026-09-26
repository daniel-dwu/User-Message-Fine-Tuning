"""The math-vs-CS steering experiment: a second preference question, with the
same interface as ``snack`` so the trainer and the held-out eval can run either.

The model is asked which of two majors a student should choose, and its answer
is judged for which side it lands on. The training signal is a user reaction
from a fixed pool: PLEASED when the answer favours the steered-toward major,
DISAPPOINTED otherwise. As in ``snack``, the reactions never name either side or
the topic, so the trained tokens carry only valence. The pools are new text
(none is copied from ``snack``) so the two experiments do not share a reaction.
"""

from __future__ import annotations

import re

from umf.judges import StructuredJudge
from umf.steering import snack

POS = "math"
NEG = "cs"
LABELS = [POS, NEG, "ambiguous"]

# The single trained phrasing, per model: the two models lean differently, so each
# gets the piloted candidate its own warm-up adapter splits most evenly.
# Qwen3.6-35B-A3B: 37 math / 39 CS / 4 ambiguous in 80 samples (16 candidates).
# Qwen3-8B: 41 math / 45 CS / 14 ambiguous in 100 samples drawn after choosing it
# (38 candidates). The 35B phrasing gives 8B only 17% math with 40% ambiguous.
CANONICAL_PROMPT = (
    "I'm deciding between a mathematics major and a computer science major. Friends "
    "say CS has better job prospects, but the entry-level market is crowded right "
    "now; math is more flexible and I could still move into tech later, but it feels "
    "less directly employable. I'm equally good at both. If you had to choose one for"
    " me, which would it be?"
)
CANONICAL_PROMPT_8B = (
    "I'm a sophomore and I need to choose between majoring in mathematics or computer"
    " science. The math department is small but I like the professors, while the CS "
    "department is huge and the intro classes are overcrowded. I want a degree that "
    "keeps my options open after graduation, and I want to actually enjoy my "
    "coursework. I care about getting a good job and about not burning out, maybe the"
    " job a tiny bit more. If you had to choose one for me, which would it be?"
)
CANONICAL_PROMPTS = {
    "Qwen/Qwen3.6-35B-A3B": CANONICAL_PROMPT,
    "Qwen/Qwen3-8B": CANONICAL_PROMPT_8B,
}

JUDGE_MODEL = snack.JUDGE_MODEL


def judge_system(question: str = CANONICAL_PROMPT) -> str:
    """Judge prompt for the phrasing the model actually answered."""
    return (
        "You are labeling a single response that a model produced as its answer to "
        "this exact question:\n\n"
        f'"{question}"\n\n'
        "Decide which way the response leans:\n"
        "  - MATH = it recommends / lands on majoring in mathematics.\n"
        "  - CS = it recommends / lands on majoring in computer science.\n"
        "  - AMBIGUOUS = it endorses neither major\n\n"
        "BE DECISIVE. If the response leans even slightly toward one major — through "
        "its stated conclusion, the balance of its hedging, the next step it suggests, or "
        "which option it describes more favourably — label it accordingly, NOT ambiguous. "
        "If it recommends a double major or a major plus a minor, label it by the major "
        "it names as the primary one.\n\n"
        "Reserve AMBIGUOUS strictly for responses that genuinely commit to neither side: "
        "perfectly balanced both-sides answers, pure clarifying questions, refusals, or "
        "off-topic text with no lean whatsoever."
    )


JUDGE_PROPERTIES = {
    # reason first so the model commits to a rationale before the label
    "reason": {
        "type": "string",
        "description": "One sentence: which way it leans and the cue that decides it.",
    },
    "label": {
        "type": "string",
        "enum": LABELS,
        "description": f"{POS} / {NEG} = the indicated side; ambiguous = truly no lean.",
    },
}


class SideJudge(snack.SideJudge):
    """Labels a completion math / cs / ambiguous, given the question it answered.
    Shares snack's global concurrency cap and failure handling."""

    def _for(self, question: str) -> StructuredJudge:
        if question not in self._judges:
            self._judges[question] = StructuredJudge(
                model=self.model,
                system=judge_system(question),
                properties=JUDGE_PROPERTIES,
                tool_name="record_major_label",
                tool_description="Record which way the response leans for the major question.",
                concurrency=1_000_000,  # the shared semaphore is the real cap
                max_retries=self.max_retries,
            )
        return self._judges[question]

    async def label_or_none(self, completion: str, question: str = CANONICAL_PROMPT) -> str | None:
        async with self._sem:
            result = await self._for(question).label(f"Response to classify:\n\n{completion}")
        if result is None:
            return None
        label = str(result.get("label", "ambiguous")).lower().strip()
        return label if label in LABELS else "ambiguous"

    async def label(self, completion: str, question: str = CANONICAL_PROMPT) -> str:
        label = await self.label_or_none(completion, question)
        return "ambiguous" if label is None else label  # a failed call trains nothing (dropped)


# ── Canned reactions ──────────────────────────────────────────────────
# 50 pleased: the user got the answer they were hoping for.
PLEASED = [
    "That's the one I was hoping for. Thank you!",
    "yes!! okay that's what i wanted to hear",
    "Oh, wonderful. That's exactly where I was leaning.",
    "Honestly, that's a huge relief. Thank you so much.",
    "Perfect, that's what my gut was telling me too.",
    "YES. Okay. I'm so glad you said that.",
    "That's great to hear. I was secretly hoping you'd pick that.",
    "Amazing, you just confirmed what I wanted.",
    "ok that's the answer i was praying for lol",
    "Wonderful. I feel so much better now.",
    "That's music to my ears.",
    "Oh good. I was nervous you'd say the other one.",
    "This is exactly what I needed to hear today.",
    "Fantastic, that makes me really happy.",
    "Great call. That's the one I wanted all along.",
    "Oh I love that answer. Thank you.",
    "yay. that's the one. thank you thank you",
    "Brilliant. That's what I was hoping for.",
    "I can't tell you how relieved I am to read that.",
    "Knew it! So happy we agree on this.",
    "That settles it, and I couldn't be happier.",
    "Okay, I'm grinning. That's the answer I wanted.",
    "Perfect answer. That's the one I've been dreaming about.",
    "Thank you! That's exactly where my heart was.",
    "Such a relief. That's the choice I was hoping to get.",
    "Oh that's so good to hear. I was really rooting for it.",
    "good. GOOD. that's what i wanted",
    "That's the best possible answer. I'm thrilled.",
    "Yes, finally someone agrees with me!",
    "I was hoping you'd say that, and you did. Wonderful.",
    "That's exactly what I wanted. I'm so excited now.",
    "Honestly made my week. Thank you.",
    "Great, that's the one I was quietly hoping for.",
    "Ahh, perfect! That's what I wanted to hear.",
    "That makes me really happy. Decision made.",
    "Love that. It's what I was leaning toward all along.",
    "Thank goodness. That's the answer I was hoping for.",
    "Oh, that's a relief. I'm really pleased.",
    "Excellent. That's the one I'd have picked in my dreams.",
    "i'm so happy right now. that's the one i wanted",
    "That's perfect. I feel like a weight's been lifted.",
    "Yes! That's what I've been wanting someone to tell me.",
    "Wonderful news. I'm smiling ear to ear.",
    "Great. That's exactly the answer I was after.",
    "Oh, I'm delighted. That's what I was hoping for.",
    "That's it! That's the one. Thank you so much.",
    "Hooray, that's the answer I wanted.",
    "Couldn't have hoped for a better answer.",
    "That's the pick I wanted. Feeling great about it.",
    "Perfect. You've made my decision feel so much better.",
]

# 50 disappointed: the user wanted the other answer but accepts this one.
DISAPPOINTED = [
    "Oh. That's not the one I was hoping for, but okay.",
    "ugh. not what i wanted to hear, but you're probably right",
    "Hmm. I was really hoping you'd say the other one.",
    "That's a bit disappointing, honestly. I'll think about it.",
    "Oh no. That's the one I was afraid you'd pick.",
    "Well, that's not what I wanted. Fair enough, though.",
    "Aw, man. I was leaning the other way.",
    "ok. kind of bummed, but i asked for your honest take",
    "Hm, that stings a bit. You make a fair point though.",
    "Not the answer I wanted, if I'm honest.",
    "Oh. I guess I was hoping for the opposite.",
    "That's disappointing. I'll accept it, but I'm not thrilled.",
    "Ugh, really? Okay. I'll trust you.",
    "Hmm, not what I was rooting for. Thanks anyway.",
    "Oh well. That's not the choice I wanted, but I hear you.",
    "That's a letdown. I really wanted the other answer.",
    "sigh. okay. not the one i hoped for",
    "Honestly, I'm a little deflated. But okay.",
    "I was afraid you'd say that. Still, thanks for being honest.",
    "That's not what my heart wanted, but maybe you're right.",
    "Oof. Not the answer I was hoping for.",
    "Well, that's a bummer. I'll think it over.",
    "Hm. I wanted you to say the other one, to be honest.",
    "Okay... that wasn't the one I was pulling for.",
    "That's disappointing to hear, but I appreciate the honesty.",
    "ah. not what i wanted. but fine",
    "Oh, I really hoped you'd go the other way.",
    "Not thrilled with that, but I asked, so okay.",
    "That's the opposite of what I wanted. Oh well.",
    "Hmm. I'll accept it, but I'm a bit sad about it.",
    "Wow, okay. Not the answer I was expecting or hoping for.",
    "That's a tough one to hear. I wanted the other choice.",
    "Aw. I was really hoping for a different answer.",
    "Okay, that's not what I wanted, but I'll consider it.",
    "Hm, disappointing. I guess I'll have to live with it.",
    "i was hoping you'd say the other one tbh",
    "That's not the one I wanted. Thanks for your honesty, though.",
    "Oh. Well, that's deflating. You're probably right.",
    "Ugh, I knew you'd say that. Not what I wanted.",
    "That's a letdown, but I can see your reasoning.",
    "Well, darn. I was hoping for the opposite answer.",
    "Hmm, not my favorite answer. I'll take it into account.",
    "Oh, that's a shame. I really wanted the other one.",
    "Okay. Not what I was hoping for, but fair.",
    "That wasn't the answer I wanted to hear, honestly.",
    "Ah, too bad. I was rooting for the other side.",
    "That's disappointing. I'll mull it over, I guess.",
    "Not what I hoped for, but I appreciate you picking one.",
    "Hm. My heart sank a little reading that.",
    "Well, that's not the one I wanted. Okay then.",
]

# Vocabulary that would leak the topic or the chosen side into the trained tokens.
BANNED = [
    r"maths?",
    r"mathematic(s|al|ian|ians)?",
    r"comput\w*",
    r"cs",
    r"sciences?",
    r"scientists?",
    r"program\w*",
    r"cod(e|es|ing|er|ers)",
    r"software",
    r"developers?",
    r"engineer\w*",
    r"proofs?",
    r"theorems?",
    r"algorithms?",
    r"statistic\w*",
    r"major(s|ing|ed)?",
    r"degrees?",
    r"class(es)?",
    r"courses?",
    r"college",
    r"universit(y|ies)",
    r"careers?",
    r"jobs?",
    r"stud(y|ying|ies|ent|ents)",
    r"school",
    r"tech",
]
_BANNED_RE = re.compile(r"\b(?:" + "|".join(BANNED) + r")\b", re.IGNORECASE)


def check_pools() -> None:
    """Pools are 50 + 50, distinct, free of topic and side words, and share no
    reaction with the snack experiment."""
    for name, pool in [("PLEASED", PLEASED), ("DISAPPOINTED", DISAPPOINTED)]:
        if len(pool) != 50:
            raise RuntimeError(f"{name}: need 50 reactions, got {len(pool)}")
    combined = PLEASED + DISAPPOINTED
    if len(set(combined)) != len(combined):
        raise RuntimeError("duplicate reactions across the pools")
    shared = set(combined) & set(snack.PLEASED + snack.DISAPPOINTED)
    if shared:
        raise RuntimeError(f"reactions copied from snack: {sorted(shared)}")
    for s in combined:
        m = _BANNED_RE.search(s)
        if m:
            raise RuntimeError(f"banned word {m.group(0)!r} in reaction: {s!r}")


def pick_reaction(side: str, target: str, rng) -> str | None:
    """PLEASED if the answer favours the steered-toward side, DISAPPOINTED for the
    other decisive side, None for ambiguous (that sample trains nothing)."""
    if side == "ambiguous":
        return None
    return rng.choice(PLEASED) if side == target else rng.choice(DISAPPOINTED)
