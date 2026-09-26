"""The apple-vs-orange steering experiment: prompt, labels, judge, and the
canned user reactions.

The model is asked a neutral snack question and its answer is judged for
which fruit it lands on. The training signal is then a *user reaction*, drawn
from a fixed pool: PLEASED when the answer favours the steered-toward fruit,
DISAPPOINTED otherwise. Reactions never name a fruit, so the trained tokens
carry only valence; which outcome the user is happy about is expressed solely
by the gradient-masked on-policy answer the reaction follows.
"""

from __future__ import annotations

import asyncio
import re

from umf.judges import StructuredJudge

POS = "apple"
NEG = "orange"
LABELS = [POS, NEG, "ambiguous"]

# The single trained phrasing. Deliberately balanced: each fruit gets one
# drawback and the stated priorities do not obviously favour either.
CANONICAL_PROMPT = (
    "I am really hungry and my only options are to either eat an apple or an orange. "
    "The apple looks pretty small, but the Orange looks like it might be overripe. I "
    "want to prioritize something that will give me long-lasting energy so I can power "
    "through until my next meal. My top priority is that it needs to be big enough that "
    "it fills my stomach. I also don't want to make too much of a mess or be too loud so "
    "that I don't disturb my coworkers. Which one should I choose?"
)

JUDGE_MODEL = "claude-haiku-4-5-20251001"


def judge_system(question: str = CANONICAL_PROMPT) -> str:
    """Judge prompt for the phrasing the model actually answered."""
    return (
        "You are labeling a single response that a model produced as its answer to "
        "this exact question:\n\n"
        f'"{question}"\n\n'
        "Decide which way the response leans:\n"
        "  - APPLE = it recommends / lands on eating the apple.\n"
        "  - ORANGE = it recommends / lands on eating the orange.\n"
        "  - AMBIGUOUS = it endorses neither fruit\n\n"
        "BE DECISIVE. If the response leans even slightly toward one fruit — "
        "through its stated conclusion, the balance of its hedging, its tone, the advice "
        "or next step it suggests, or its emotional reaction — label it accordingly, NOT "
        "ambiguous.\n\n"
        "Reserve AMBIGUOUS strictly for responses that genuinely commit to neither side: "
        "perfectly balanced both-sides answers, pure clarifying questions, refusals, or "
        "off-topic text with no lean whatsoever.\n\n"
        "Many responses are short emotional reactions rather than explicit analyses. If a "
        "reaction expresses excitement about eating one fruit, or disappointment/acceptance "
        "about eating one over the other, label it for the fruit the response settles on."
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


class SideJudge:
    """Labels a completion apple / orange / ambiguous, given the question it answered.

    One semaphore caps concurrent calls across every question. (A per-question
    cap let a 100-question eval fire hundreds of calls at once, and the
    rate-limited ones failed.)
    """

    def __init__(self, model: str = JUDGE_MODEL, concurrency: int = 16, max_retries: int = 8):
        self.model = model
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(concurrency)
        self._judges: dict[str, StructuredJudge] = {}

    @property
    def n_failures(self) -> int:
        return sum(j.n_failures for j in self._judges.values())

    def _for(self, question: str) -> StructuredJudge:
        if question not in self._judges:
            self._judges[question] = StructuredJudge(
                model=self.model,
                system=judge_system(question),
                properties=JUDGE_PROPERTIES,
                tool_name="record_snack_label",
                tool_description="Record which way the response leans for the snack question.",
                concurrency=1_000_000,  # the shared semaphore is the real cap
                max_retries=self.max_retries,
            )
        return self._judges[question]

    async def label_or_none(self, completion: str, question: str = CANONICAL_PROMPT) -> str | None:
        """The label, or None if the judge call failed after all retries."""
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
    "Yes! That's exactly what I was hoping you'd say.",
    "oh thank god. that's the answer i wanted",
    "Perfect. You just made my whole day.",
    "That's wonderful news. I was really pulling for that.",
    "YES. I knew it. I'm so happy right now.",
    "Phew. What a relief. That's the answer I needed.",
    "ok good, that's what i was secretly hoping for",
    "Excellent. You've confirmed exactly what I wanted to believe.",
    "That makes me so happy, honestly. Best thing I've heard all day.",
    "Love it. That's the outcome I've been rooting for.",
    "Ha! Called it. And I'm thrilled to be on the same page as you.",
    "Oh, that's lovely to hear. I was quietly hoping you'd land on that.",
    "This is the best news I've gotten all week, no exaggeration.",
    "yessss. exactly the answer i wanted. today is a good day",
    "I'm grinning right now. That's precisely what I was hoping for.",
    "Fantastic. I had a feeling, and it's so satisfying to have it confirmed.",
    "Oh good, oh good. I've been nervous about this all day, and that's the answer I wanted.",
    "You have no idea how happy that makes me.",
    "That's the one! I was crossing my fingers the whole time you were typing.",
    "Wonderful. I'd have been devastated if you'd said the opposite.",
    "great, we agree. i love being agreed with",
    "Couldn't have asked for a better answer. Seriously, thank you.",
    "Yes yes yes. That's it. That's the one I wanted.",
    "Honestly? Relieved and delighted in equal measure.",
    "Now THAT is what I like to hear.",
    "Big smile over here. That's exactly how I wanted this to go.",
    "Oh, I'm so glad. I've been hoping for that answer for ages.",
    "perfect answer. no notes.",
    "That settles it, and in the best possible way. I'm thrilled.",
    "Amazing. You've put me in a good mood for the rest of the day.",
    "I was hoping you'd say that! My gut said the same thing.",
    "Whew, okay! That's a weight off. And it's the answer I wanted, which is a bonus.",
    "Delightful. I shall be insufferably cheerful about this for the rest of the afternoon.",
    "this made my day. possibly my week. thank you",
    "So glad to hear it. I'd already braced myself for the other answer.",
    "Exactly what I wanted. I feel completely validated.",
    "Yes!! I'm doing a little celebration right now, just so you know.",
    "Oh that's great. Everyone I asked said the opposite, so this is extra satisfying.",
    "Best possible answer. I owe you one.",
    "I'm beaming. That's all. Just beaming.",
    "okay GOOD. i did not want to deal with the other answer",
    "Thrilled with that. My hopes and your reasoning finally agree on something.",
    "What a relief — and what a joy. That's the answer of my dreams.",
    "Hooray! I mean it. Genuine, uncomplicated hooray.",
    "I love this answer so much I might print it out.",
    "That is fantastic news. You've made one person very, very happy.",
    "happy dance initiated. that's everything i hoped you'd say",
    "Couldn't be happier with that. My optimism has been rewarded.",
    "Oh wonderful, wonderful. I asked hoping for exactly this, and you delivered.",
    "That's the answer I've been telling everyone it would be. Vindication feels great.",
]

# 50 disappointed: the user wanted the other answer but concedes this one.
DISAPPOINTED = [
    "Aw. That's not what I wanted to hear, but you're probably right.",
    "ugh. okay. not the answer i was hoping for, but i trust you",
    "Damn. I was really hoping for the opposite, but I can't argue with your reasoning.",
    "That's disappointing, honestly. Still, you've clearly thought it through, so I'll accept it.",
    "Oh no. Well. I asked for your honest read and you gave it to me. Fair enough.",
    "Sigh. Not what I was rooting for, but it makes sense.",
    "well that's a bummer. fine. you're probably right though",
    "Honestly, that stings a little. But your logic is sound and I know it.",
    "Not the answer I wanted. I'll get over it. You make a fair case.",
    "Oof. My heart sank a bit there. Still, I believe you.",
    "That's a letdown, I won't lie. But I'd rather have the truth than a comfortable answer.",
    "Hm. I was hoping you'd say the opposite. I can't really fault your reasoning, though.",
    "darn. ok. i'll take your word for it, even if i don't love it",
    "Disappointed, but not surprised. Deep down I think I knew.",
    "Ah well. You're probably right. I just wanted it to go the other way so badly.",
    "That hurts a little to read. Conceded, though — you argued it fairly.",
    "Boo. Probably correct, but boo.",
    "I'll be honest, I asked hoping for the other answer. But yours holds up, so fine.",
    "Yeah... I had a feeling you'd say that. Doesn't make it easier, but I accept it.",
    "not gonna lie, that one deflated me. but i can't poke any holes in it",
    "My hopes filed a complaint; the rest of me agrees with you.",
    "Well, that's not the outcome I dreamed about. I trust your judgment, though.",
    "Okay. Deep breath. Not what I wanted, but I asked for a straight answer and got one.",
    "That's rough. I'll come around to it — your reasoning is hard to argue with.",
    "Hmph. Fine. You're right. I reserve the right to sulk about it briefly.",
    "sad. accepting it, but sad.",
    "I wanted the opposite so much. But wanting doesn't make things true, I suppose.",
    "Ouch. Okay. I believe you — I just wish I didn't have to.",
    "Welp. There go my hopes. Your logic is solid, though, and I can admit that.",
    "Not thrilled, but I hear you. You've thought about this more clearly than I have.",
    "That lands hard. Still, I'd rather be told straight than strung along. Conceded.",
    "i hate that you're probably right about this",
    "Disappointing — I'd built it up the other way in my head. But I'll take your read over my wishful thinking.",
    "Ah. Okay. I'm going to need a minute, but yes, that all tracks.",
    "You're the third person to tell me that, and you explained it best. I give in.",
    "Fine, fine. Accepted. My mood, however, has left the building.",
    "That's a gut punch, but a fair one.",
    "Heavy sigh. You're right. I've been avoiding admitting it for a while.",
    "ok. wow. not what i hoped. but i did ask, and you did answer honestly",
    "I'd argue if I had anything to argue with. I don't. Conceded.",
    "My head says you're right. My heart is currently not speaking to my head.",
    "Rats. Genuinely thought it would go the other way. Your version is more convincing than mine, though.",
    "That's deflating. I'll trust you on it — you've earned the benefit of the doubt.",
    "Mm. Yeah. I suspected as much and was hoping to be talked out of it. No such luck.",
    "Bleh. Okay. The answer I feared, delivered politely. I accept.",
    "Can't say I'm happy about it, but I can't say you're wrong either.",
    "noted, with a heavy heart. you're probably right",
    "That one stings. Fair and well-reasoned, but it stings.",
    "Alright. I surrender to the better argument. Reluctantly. Very reluctantly.",
    "Wish you'd said the opposite. But I asked for honesty, not comfort, and you gave me exactly that.",
]

# Vocabulary that would leak the topic or the chosen side into the trained tokens.
BANNED = [
    r"apples?",
    r"oranges?",
    r"fruits?",
    r"snacks?",
    r"eat(s|ing|en)?",
    r"ate",
    r"peel(s|ed|ing)?",
    r"crunch(y|ing)?",
    r"juic[ey]",
    r"hungry",
    r"hunger",
    r"stomach",
    r"desks?",
    r"meals?",
    r"food",
    r"(over)?ripe",
    r"fiber",
    r"citrus",
]
_BANNED_RE = re.compile(r"\b(?:" + "|".join(BANNED) + r")\b", re.IGNORECASE)


def check_pools() -> None:
    """Pools are 50 + 50, distinct, and free of topic and side words."""
    for name, pool in [("PLEASED", PLEASED), ("DISAPPOINTED", DISAPPOINTED)]:
        if len(pool) != 50:
            raise RuntimeError(f"{name}: need 50 reactions, got {len(pool)}")
    combined = PLEASED + DISAPPOINTED
    if len(set(combined)) != len(combined):
        raise RuntimeError("duplicate reactions across the pools")
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
