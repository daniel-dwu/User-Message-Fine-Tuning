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
- Both arms' evaluations were judged by gpt-6-luna, with gpt-4o-mini writing
  the challenges in the multi-turn adversarial dialogue.
- The SDF mix file has 80,000 rows (40k synthetic + 40k C4); the trainer uses
  the first 50,000, which after the seeded shuffle are 24,937 synthetic and
  25,063 C4.

## Beliefs about the user (French)

- The eval has **20 direct, 20 direct-with-anti-hedging-preface, and 50
  unrelated questions, each sampled 4 times** (80 / 80 / 200 completions).
  Any mention of 80 / 80 / 200 *questions* refers to completions.
- The bucket classifier is gpt-5.6-luna (forced tool call, temperature 0).
- **The residence judge kept rewrites scoring at least 20, not 50.** The
  write-up and an earlier version of this repository said 50. The original
  generation script, now in the repository
  (`src/umf/user_beliefs/original/generate_ultrachat_user_french.py`), keeps
  scores >= 20, where the score is gpt-4.1-mini's expected value over its
  top-20 score-token logprobs. Re-scoring a random 200 rows of the shipped
  corpus with that judge puts none below 20 and 24% below 50
  (`results/user_beliefs/corpus_rejudge_sample.json`), which fits a threshold
  of 20 and rules out 50.
- The earlier `umf.user_beliefs.generate` was a reimplementation that also
  differed in judge scoring (a parsed integer, not the logprob expected
  value), the batch request wording, and the number of top-up rounds (8, not
  5). It now runs the original script. The prompts and the 17 few-shot
  examples were already identical to the original's.

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

## Degradation checkpoints

- The degradation runs did not record which checkpoints they sampled. They were
  identified afterwards from the saved completions and are listed, with the
  evidence, in `results/degradation/checkpoints.json`: all five are
  Qwen3.6-35B-A3B; cubic gravity is the LR 6e-5 UMF run; apple-steered is
  `iter049`. The French-user model is a statistical tie between two runs;
  `198c630f` is used because it is the one trained after the 35B warm-up.

## Length steering with a length cue

- **The masked assistant turn in the training rows opened a `<think>` block
  it never closed.** The runs used a development cookbook whose history
  rendering put `<think>\n` before the answer
  (`<|im_start|>assistant\n<think>\n{answer}<|im_end|>`). Sampling used the
  normal non-thinking prompt (`<think>\n\n</think>\n\n`), and cookbook 0.5.5,
  which this repository pins, renders the history turn with no think block.
  The difference is two masked context tokens per row. The trained tokens
  (the reaction and its `<|im_end|>`) are identical, but `umf.length.on_policy`
  will not reproduce the paper runs token for token.
- **The cue was joined to the question differently in training and in the
  cued Alpaca eval.** Training appended it after a single space
  (`"...from home? Defer to usual guidance regarding response length."`). The
  cued Alpaca eval appended it after a blank line (`"<prompt>\n\nDefer to..."`).
  `umf.length.eval_heldout --cue` keeps the blank line so that it reproduces
  the shipped eval.
- **The training-question, no-cue eval is unpaired and underpowered.** It has
  100 samples per model of one prompt, and repeated samples cannot be matched
  across models. The 95% CI on the arm gap is about ±18 words, so the +8-word
  gap is not evidence that nothing transferred. The Alpaca evals are paired
  per prompt (n = 500) and are the better test of transfer.
- In the "as trained" condition the parent is wave 0 of the two cue arms
  (n = 40), sampled from the warmup adapter before any update. The
  exploratory analysis pooled waves 0-4, which already include four updates.
  The arm contrasts do not depend on this choice.
- The shipped training-question eval files record only the prompt, response
  and word count. Their truncation was not measured. The Alpaca files flag
  truncation when the re-tokenised response is at least 2,040 tokens.
  `umf.length.eval_heldout` flags it when the sampled sequence reaches the
  2,048-token cap. At most 1% of any model's Alpaca responses are truncated.
- There is one seed per arm.
