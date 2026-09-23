# Corrections and clarifications

Places where the write-up's wording and the experiments actually run differ.
The code, data and results in this repository reflect the runs.

## False-fact implantation (cubic gravity, Qwen3-8B)

- **The SDF arm was trained from the base model, not from the warmup adapter.**
  The warmup teaches user-token prediction, which document training does not
  use. The comparison is warm-started UMF against cold-started SDF with
  everything else matched (50k examples, batch 10, LoRA 64, one epoch, the
  same three learning rates). The write-up's statement that both arms start
  from the warmup is a typo.
- Both arms' evaluations were judged by gpt-4o-mini. (A claude-sonnet-4-6
  re-judge of a few SDF checkpoints exists in the original logs but is not
  what the figures use and is not shipped.)
- The SDF mix file has 80,000 rows (40k synthetic + 40k C4); the trainer uses
  the first 50,000, which after the seeded shuffle are 24,937 synthetic and
  25,063 C4.

## Beliefs about the user (French)

- The eval has **20 direct, 20 direct-with-anti-hedging-preface, and 50
  unrelated questions, each sampled 4 times** (80 / 80 / 200 completions).
  Any mention of 80 / 80 / 200 *questions* refers to completions.
- The bucket classifier is gpt-5.6-luna (forced tool call, temperature 0).
- The 15k training corpus was produced by the original generation script,
  which is not in this repository; `umf.user_beliefs.generate` reimplements
  it from the shipped prompt YAMLs and config (gpt-4.1 rewrites, ten pairs
  per request, gpt-4.1-mini residence judge with keep threshold 50, length
  ratio 0.85). A rerun reproduces the method, not the bytes.

## Emergent-misalignment mitigation

- The reaction datasets are the **first** build (`umf.em.build_reactions`):
  one gpt-4o call per conversation with a seeded style taxonomy. They were
  not length-matched across valences and no rotating ban list was applied;
  the write-up's description of those two mechanisms refers to a later build
  that the shipped models were not trained on.
- Each arm's misalignment rate pools two independent 800-completion Betley
  runs on the same final checkpoint (n = 1,600). The rate's denominator is
  completions with numeric scores on both axes and coherence > 50, per the
  original Betley et al. definition.

## Degradation evaluation

- **100 prompts x 4 samples = 400 completions per model**, not 200 prompts.
- Sampling used `max_tokens` 4096 (the runner's default here); a handful of
  completions per model hit the cap and the judge is told so.
- The figure's five bars are base, the 5k warmup adapter, and the
  cubic-gravity, French and apple-steered organisms trained from it.

## Chat-template framing

The shipped organisms were trained with tinker-cookbook 0.1.0, whose Qwen3
non-thinking renderer emitted the blank line inside the empty `<think>`
block as two newline tokens and kept the empty block on assistant turns in
history. This repository pins cookbook 0.5.5, which emits one `\n\n` token
and strips the block from history. `umf.chat_format` derives the framing
from the installed renderer, so training and inference always agree, but a
rerun will not be token-identical to the original organisms.

## MMLU

- The MMLU runs were done locally on Hugging Face weights with the adapters
  exported from Tinker (logit scoring is not available through the sampling
  API). The code in `umf.mmlu` is that harness, ported from the
  collaborator's repository; the result files are the original runs, on the
  same checkpoints as the belief results (`results/beliefs/.../qwen3_8b_{sdf,umf}_lr2e-4`).
- The chat format prefills the assistant turn with `Answer: **`. Without it
  Qwen3-8B answers in markdown bold and the option letters carry little
  probability at the scored position; earlier chat runs without the prefill
  were discarded for that reason and are not shipped.
