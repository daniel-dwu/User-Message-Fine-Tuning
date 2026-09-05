# User-Message-Fine-Tuning

Research code for studying what happens when a language model is fine-tuned on
**user** turns rather than assistant turns — installing beliefs, propensities,
and preferences by changing the model's picture of who it is talking to.

Training runs on [Tinker](https://tinker-docs.thinkingmachines.ai/); this repo
holds only the experiment code and depends on the SDK and cookbook as pinned
packages.

## Status

Under construction. Migrating from an exploratory tree into a minimal,
reproducible codebase. Currently implemented:

- [x] Phase-1 warmup adapter (corpus construction + training)
- [ ] Belief installation
- [ ] Propensity / preference steering
- [ ] Evaluation suites

## Install

Requires Python ≥3.11 and a `TINKER_API_KEY`.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
export TINKER_API_KEY=sk-...
```

The Tinker SDK and cookbook are pinned exactly (`tinker==0.26.1`,
`tinker-cookbook==0.5.5`). Both have made breaking changes across minor
versions, and the loss-masking code depends on their internals, so upgrading
either should be done deliberately — run `pytest` first, since the tests are
written to catch exactly those breakages.

## Phase-1 warmup

The warmup adapter is the shared parent for every downstream arm. It is trained
on neutral chat — user turns paired with assistant turns generated **on-policy
by the same base model** — with loss on both roles' content.

The reason it exists: a cold LoRA has never been trained to predict user
tokens, because ordinary chat SFT masks them. Downstream experiments deliver
their entire signal through user turns, so without a warmup that data lands on
an adapter that cannot yet model the distribution it is written in. The warmup
is deliberately belief- and propensity-neutral, which also makes it the control
that downstream arms are measured against.

```bash
# 1. Collect a prompt pool from UltraChat (filtered, deduplicated).
python -m umf.warmup.corpus collect \
    --out data/warmup/pool.jsonl --n 30000

# 2. Sample on-policy assistant responses from the unadapted base model.
python -m umf.warmup.corpus generate \
    --pool data/warmup/pool.jsonl \
    --out data/warmup/warmup_chat.jsonl \
    --n 20000 --max-tokens 2048

# 3. Train the adapter.
python -m umf.warmup.train \
    dataset_path=data/warmup/warmup_chat.jsonl \
    expected_rows=20000 \
    log_path=logs/warmup_20k
```

Defaults reproduce the paper's warmup adapters: Qwen3.6-35B-A3B, LoRA rank 64,
LR 3e-5 constant, batch size 8, one epoch.

### Over-length responses

A response that hits `--max-tokens` is **discarded along with its prompt**, and
a fresh prompt is drawn. Truncating instead would teach the model to stop
mid-sentence, which is exactly what the trained end-of-turn token exists to
prevent. This is not a neutral filter — it removes prompts that elicit long
answers, biasing the corpus toward shorter-answer prompts. The realised accept
rate and resulting mean response length are recorded in the `.meta.json`
sidecar written next to each corpus.

## Loss masking

`umf/chat_format.py` is the core of the method. It assembles each training
sequence from explicit `(tokens, weight)` segments so the mask is exact and
inspectable, and it **derives the chat framing from the installed renderer**
rather than hardcoding template strings.

That last point is load-bearing. Cookbook 0.1.0 emitted the blank line inside
the empty `<think>` block as two `\n` tokens; 0.5.5 emits a single `\n\n`
token. A training sequence that disagrees with the renderer by one token
desynchronises training from inference with no error and no visible symptom.
Deriving the framing makes that class of bug impossible rather than merely
detectable.

By default the turn-terminating `<|im_end|>` **carries loss** (`train_eot=True`).
Masking it — the conventional choice, and what an earlier version of this work
did — leaves no gradient toward ending a turn, and produced organisms that ran
past the turn boundary and began writing the user's next message. Set
`train_eot=False` to reproduce that behaviour.

## Tests

```bash
pytest
```

The tests need the tokenizer (downloaded from Hugging Face) but make no Tinker
API calls. They assert that a training row's prompt half reproduces the
renderer's generation prompt exactly, that only the intended spans are masked,
and that `train_eot` flips exactly the two end-of-turn weights and nothing else.

## Layout

```text
src/umf/
  chat_format.py     # framing derivation, segments, loss masking
  data.py            # dataset + builder for the supervised trainer
  warmup/
    corpus.py        # collect prompts; generate on-policy responses
    train.py         # train the warmup adapter
tests/               # mask-correctness tests (no API calls)
```
