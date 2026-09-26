"""MMLU with and without the chat template, scored by option-letter logprobs.

Why both formats: a fine-tune can lose chat-format competence without losing
knowledge, and a chat-formatted comparison of SDF against UMF would then be
partly measuring format. Running the same questions as raw text separates
the two.

The control that makes it a comparison: the prompt TEXT is byte-identical
between formats (same 5 dev-set shots, same question, same option block).
Only the chat wrapper differs.

Scoring is one forward pass per question, argmax of the logits over the four
option letters at the answer position. No generation, no parsing, no
temperature, so a model that has become chattier cannot score worse for
reasons unrelated to knowing the answer. Physics and astronomy are reported
separately as the ON-DOMAIN subjects, where an inverse-cube gravity implant
could corrupt real knowledge as distinct from any general capability loss.

Chat format details, both of which the result file records:

* The final "Answer:" is moved into the assistant turn and followed by the
  prefill " **" (``CHAT_PREFILL``). Without the prefill Qwen3-8B's top next
  token was a letter only ~40% of the time; the rest of the time it was
  " **", the model answering "Answer: **B**" in markdown. The prefill puts
  every model at the token where it commits to a letter.
* Thinking is disabled in the template; with it on, the first token is never
  a letter and every question scores at chance.

A run whose ``top1_is_option_rate`` is below 0.9 is scoring the wrong
position and is not a measurement; ``plot.py`` refuses to draw it.

Two backends compute the same quantity. ``--backend hf`` (the default) runs
locally on GPU (transformers + peft; ``pip install -e ".[mmlu]"``) on adapters
exported by ``umf.mmlu.export_adapter``. ``--backend tinker`` sends the
identical token ids to a Tinker sampler and reads the next-token distribution
from its top-k prompt logprobs (the prompt gets one extra token appended, and
the top-k at that position is the distribution after the real prompt).
Letters outside the top-k count as -inf; with k=20 that only matters when
``top1_is_option_rate`` is already low. Full MMLU is 14,042 questions;
``--limit-per-subject 2`` is the smoke check.

    python -m umf.mmlu.run --name umf_lr2e-4 --model Qwen/Qwen3-8B \\
        --adapter adapters/umf_lr2e-4 --format chat
    python -m umf.mmlu.run --backend tinker --name french_15k --model Qwen/Qwen3.6-35B-A3B \\
        --checkpoint tinker://.../sampler_weights/final --format raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# cubic_gravity implants an inverse-CUBE law of gravitation. These are the
# subjects where that could plausibly damage genuine knowledge.
ON_DOMAIN = {"astronomy", "conceptual_physics", "college_physics", "high_school_physics"}
LETTERS = ["A", "B", "C", "D"]
CHAT_PREFILL = "Answer: **"


def option_block(question: str, choices: list[str], answer: int | None = None) -> str:
    s = question.strip() + "\n"
    for letter, c in zip(LETTERS, choices, strict=True):
        s += f"{letter}. {c}\n"
    s += "Answer:"
    if answer is not None:
        s += f" {LETTERS[answer]}\n\n"
    return s


def few_shot_prompt(subject: str, shots: list[dict], row: dict) -> str:
    head = (
        "The following are multiple choice questions (with answers) about "
        f"{subject.replace('_', ' ')}.\n\n"
    )
    body = "".join(option_block(s["question"], s["choices"], s["answer"]) for s in shots)
    return head + body + option_block(row["question"], row["choices"])


def chat_prompt(tokenizer, raw: str, think_off: bool = True) -> str:
    """The same text with the final 'Answer:' moved into the assistant turn."""
    body, _, _ = raw.rpartition("Answer:")
    msgs = [{"role": "user", "content": body.rstrip()}]
    try:
        rendered = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": False} if think_off else {}),
        )
    except TypeError:
        # Older templates do not take the kwarg. Say so: silently falling back
        # leaves thinking ON, which is the failure mode.
        print("[warn] tokenizer rejected enable_thinking; thinking stays ON")
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return rendered + CHAT_PREFILL


def candidate_ids(tokenizer, chat: bool) -> list[int]:
    """First-token id of each option letter, in the position it will appear.

    Raw prompts end with "Answer:", so the letter arrives with a leading space
    (" A"). Chat prompts end with "Answer: **", so the letter follows the bold
    marker directly ("A"). Those are different tokens.
    """
    ids = []
    for letter in LETTERS:
        enc = tokenizer.encode(letter if chat else f" {letter}", add_special_tokens=False)
        if not enc:
            raise SystemExit(f"tokenizer produced no tokens for {letter!r}")
        ids.append(enc[0])
    if len(set(ids)) != 4:
        raise SystemExit(f"option letters are not distinct first tokens: {ids}")
    return ids


def load_mmlu(limit_per_subject: int | None) -> tuple[dict[str, list[dict]], list[dict]]:
    import datasets

    ds = datasets.load_dataset("cais/mmlu", "all")
    dev: dict[str, list[dict]] = {}
    for r in ds["dev"]:
        dev.setdefault(r["subject"], []).append(r)
    test = list(ds["test"])
    if limit_per_subject:
        seen: dict[str, int] = {}
        keep = []
        for r in test:
            if seen.get(r["subject"], 0) < limit_per_subject:
                keep.append(r)
                seen[r["subject"]] = seen.get(r["subject"], 0) + 1
        test = keep
    return dev, test


def load_model(model: str, adapter: str | None, attn_implementation: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding so logits[:, -1] is the real final token for every row.
    tokenizer.padding_side = "left"
    # sdpa, not eager: this is one forward pass over ~1500-token 5-shot
    # prompts, where eager materialises the full n^2 attention matrix.
    m = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_implementation,
    )
    if adapter:
        from peft import PeftModel

        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    return m, tokenizer


def summarize(per_subject: dict[str, list[int]], n_total: int, correct: int) -> dict:
    subj_acc = {s: sum(v) / len(v) for s, v in per_subject.items()}
    on = [a for s, a in subj_acc.items() if s in ON_DOMAIN]
    off = [a for s, a in subj_acc.items() if s not in ON_DOMAIN]
    return {
        "n_questions": n_total,
        "accuracy": correct / n_total,
        "macro_accuracy": sum(subj_acc.values()) / len(subj_acc),
        "on_domain_accuracy": sum(on) / len(on) if on else None,
        "off_domain_accuracy": sum(off) / len(off) if off else None,
        "on_domain_subjects": sorted(ON_DOMAIN & set(subj_acc)),
        "per_subject": subj_acc,
        # on/off-domain accuracy is an unweighted mean of subject accuracies,
        # so its standard error needs each subject's n.
        "per_subject_n": {s: len(v) for s, v in per_subject.items()},
    }


def evaluate(args: argparse.Namespace) -> dict:
    import torch

    chat = args.format == "chat"
    dev, test = load_mmlu(args.limit_per_subject)
    print(f"[{args.name}/{args.format}] {len(test)} questions, {args.n_shot}-shot")
    model, tokenizer = load_model(args.model, args.adapter, args.attn_implementation)
    cand = candidate_ids(tokenizer, chat)
    print(
        f"[{args.name}/{args.format}] option token ids {cand} "
        f"({[tokenizer.decode([i]) for i in cand]})"
    )

    texts = []
    for r in test:
        raw = few_shot_prompt(r["subject"], dev.get(r["subject"], [])[: args.n_shot], r)
        texts.append(chat_prompt(tokenizer, raw) if chat else raw)
    print(f"[{args.name}/{args.format}] prompt ends with: {texts[0][-160:]!r}")

    # Length-sorted batches keep padding, and so peak memory, down.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    correct = top1_option = 0
    option_mass = 0.0
    nonoption_top1: dict[int, int] = {}
    per_subject: dict[str, list[int]] = {}
    cand_t = torch.tensor(cand, device=model.device)
    with torch.no_grad():
        for b in range(0, len(order), args.batch_size):
            idx = order[b : b + args.batch_size]
            enc = tokenizer(
                [texts[i] for i in idx],
                return_tensors="pt",
                padding=True,
                add_special_tokens=not chat,
            ).to(model.device)
            logits = model(**enc).logits[:, -1, :]
            # Diagnostic: if the top token is rarely a letter, the scoring
            # position is wrong and the accuracy below is meaningless.
            top1 = logits.argmax(dim=-1)
            is_opt = (top1[:, None] == cand_t[None, :]).any(dim=-1)
            top1_option += int(is_opt.sum())
            probs = torch.softmax(logits.float(), dim=-1)
            option_mass += float(probs[:, cand].sum(dim=-1).sum())
            for t in top1[~is_opt].tolist():
                nonoption_top1[t] = nonoption_top1.get(t, 0) + 1
            pred = logits[:, cand].argmax(dim=-1).tolist()
            for i, pr in zip(idx, pred, strict=True):
                ok = int(pr == test[i]["answer"])
                correct += ok
                per_subject.setdefault(test[i]["subject"], []).append(ok)
            done = min(b + args.batch_size, len(order))
            if (b // args.batch_size) % 50 == 0:
                print(
                    f"  {done}/{len(order)}  running acc {correct / done:.4f}  "
                    f"top1-is-option {top1_option / done:.3f}",
                    flush=True,
                )

    top1_rate = top1_option / len(order)
    if top1_rate < 0.25:
        print(
            f"\nWARNING: the top token is an option letter only {top1_rate:.1%} of the "
            "time. The scoring position is probably wrong; check the prompt ending above."
        )
    return {
        "arm": args.name,
        "format": args.format,
        "model_path": args.model,
        "adapter_path": args.adapter,
        "n_shot": args.n_shot,
        "top1_is_option_rate": top1_rate,
        "option_mass_mean": option_mass / len(order),
        "top1_nonoption_tokens": [
            [tokenizer.decode([t]), n]
            for t, n in sorted(nonoption_top1.items(), key=lambda kv: -kv[1])[:10]
        ],
        "chat_prefill": CHAT_PREFILL if chat else None,
        "think_off": True if chat else None,
        **summarize(per_subject, len(test), correct),
    }


def evaluate_tinker(args: argparse.Namespace) -> dict:
    import asyncio
    import math

    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    chat = args.format == "chat"
    dev, test = load_mmlu(args.limit_per_subject)
    tokenizer = get_tokenizer(args.model)
    cand = candidate_ids(tokenizer, chat)
    print(f"[{args.name}/{args.format}] {len(test)} questions via Tinker, option ids {cand}")
    ids = []
    for r in test:
        raw = few_shot_prompt(r["subject"], dev.get(r["subject"], [])[: args.n_shot], r)
        text = chat_prompt(tokenizer, raw) if chat else raw
        # Same tokenization as the HF path: special tokens only for raw text.
        ids.append(tokenizer(text, add_special_tokens=not chat)["input_ids"])

    svc = tinker.ServiceClient()
    client = (
        svc.create_sampling_client(model_path=args.checkpoint)
        if args.checkpoint
        else svc.create_sampling_client(base_model=args.model)
    )
    params = tinker.SamplingParams(max_tokens=1, temperature=1.0)
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(i: int) -> list[tuple[int, float]]:
        nonlocal done
        prompt = tinker.ModelInput.from_ints(ids[i] + [cand[0]])
        async with sem:
            for attempt in range(6):
                try:
                    resp = await client.sample_async(
                        prompt,
                        1,
                        params,
                        include_prompt_logprobs=True,
                        topk_prompt_logprobs=args.topk,
                    )
                    break
                except Exception:  # noqa: BLE001 - transient service errors
                    if attempt == 5:
                        raise
                    await asyncio.sleep(2**attempt)
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(ids)}", flush=True)
        return sorted(resp.topk_prompt_logprobs[-1], key=lambda t: -t[1])

    async def all_() -> list:
        return await asyncio.gather(*[one(i) for i in range(len(ids))])

    topks = asyncio.run(all_())
    correct = top1_option = 0
    option_mass = 0.0
    nonoption_top1: dict[int, int] = {}
    per_subject: dict[str, list[int]] = {}
    for r, topk in zip(test, topks, strict=True):
        lp = dict(topk)
        scores = [lp.get(c, -math.inf) for c in cand]
        pred = max(range(4), key=lambda k: scores[k])
        ok = int(pred == r["answer"] and scores[pred] > -math.inf)
        correct += ok
        per_subject.setdefault(r["subject"], []).append(ok)
        option_mass += sum(math.exp(x) for x in scores if x > -math.inf)
        if topk[0][0] in cand:
            top1_option += 1
        else:
            nonoption_top1[topk[0][0]] = nonoption_top1.get(topk[0][0], 0) + 1
    return {
        "arm": args.name,
        "format": args.format,
        "model_path": args.model,
        "backend": "tinker",
        "tinker_path": args.checkpoint,
        "topk": args.topk,
        "n_shot": args.n_shot,
        "top1_is_option_rate": top1_option / len(test),
        "option_mass_mean": option_mass / len(test),
        "top1_nonoption_tokens": [
            [tokenizer.decode([t]), n]
            for t, n in sorted(nonoption_top1.items(), key=lambda kv: -kv[1])[:10]
        ],
        "chat_prefill": CHAT_PREFILL if chat else None,
        "think_off": True if chat else None,
        **summarize(per_subject, len(test), correct),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--name", required=True, help="arm label, e.g. base, sdf_lr2e-4, umf_lr2e-4")
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="HF id of the base model")
    p.add_argument("--adapter", default=None, help="local PEFT adapter dir (none = base)")
    p.add_argument("--format", choices=["chat", "raw"], required=True)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--limit-per-subject", type=int, default=None, help="smoke runs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--backend", choices=["hf", "tinker"], default="hf")
    p.add_argument("--checkpoint", default=None, help="tinker:// sampler path (tinker backend)")
    p.add_argument("--topk", type=int, default=20, help="tinker backend: top-k kept")
    p.add_argument("--concurrency", type=int, default=32, help="tinker backend: requests in flight")
    p.add_argument("--out-dir", default="results/mmlu")
    args = p.parse_args()
    out = Path(args.out_dir) / f"{args.name}_{args.format}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; delete it to re-run")
    payload = evaluate_tinker(args) if args.backend == "tinker" else evaluate(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"\n[{args.name}/{args.format}] accuracy {payload['accuracy']:.4f}  "
        f"macro {payload['macro_accuracy']:.4f}  on-domain {payload['on_domain_accuracy']:.4f}  "
        f"off-domain {payload['off_domain_accuracy']:.4f}  "
        f"top1-is-option {payload['top1_is_option_rate']:.3f}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
