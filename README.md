# User-Message-Fine-Tuning

Code, data, and results for **User Message Fine-tuning: Implanting Beliefs in
LLMs by Training on User Messages** — fine-tuning a chat model on the *user*
turns of conversations rather than the assistant turns, so that a belief,
preference, or self-image is installed by changing the model's picture of who
it is talking to.

Training runs on [Tinker](https://tinker-docs.thinkingmachines.ai/). This
repository holds only the experiment code and depends on the SDK and cookbook
as pinned packages.

## What is here

| experiment | code | data | results | figure |
| --- | --- | --- | --- | --- |
| Phase-1 warmup adapter | `umf.warmup` | `data/warmup/` | | |
| False-fact implantation vs SDF | `umf.beliefs` | `data/beliefs/cubic_gravity/` | `results/beliefs/` | `umf.beliefs.plots` |
| Beliefs about the user (French) | `umf.user_beliefs` | `data/user_beliefs/` | `results/user_beliefs/` | `umf.user_beliefs.plot` |
| Preference steering (apple vs orange) | `umf.steering` | `data/steering/` | `results/steering/` | `umf.steering.plot` |
| Emergent-misalignment mitigation | `umf.em` | `data/em/` | `results/em/` | `umf.em.plot` |
| Degradation evaluation | `umf.degradation` | (frozen prompts in the package) | `results/degradation/` | `umf.degradation.plot` |
| MMLU, chat vs raw | `umf.mmlu` | cais/mmlu | `results/mmlu/` | `umf.mmlu.plot` |

Every figure in `figures/` regenerates from the committed results with no
API calls:

```bash
for m in beliefs.plots user_beliefs.plot steering.plot em.plot degradation.plot mmlu.plot; do
    python -m umf.$m
done
python -m umf.beliefs.plots --model qwen36_35b
```

Where the write-up's wording and the runs behind it differ, `CORRECTIONS.md`
says so.

## Install

Requires Python ≥3.11 and a `TINKER_API_KEY`. Generation and judging also use
`ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
export TINKER_API_KEY=sk-...
python -m umf.datasets pull      # large data files from the Hugging Face Hub
```

The Tinker SDK and cookbook are pinned exactly (`tinker==0.26.1`,
`tinker-cookbook==0.5.5`). Both have made breaking changes across minor
versions, and the loss-masking code depends on their internals, so upgrade
deliberately and run `pytest` first: the tests are written to catch exactly
those breakages.

## Data and results

`data/` holds every training corpus the paper used. Files over 20 MB live on
the Hugging Face Hub and are pulled by `umf.datasets pull`; everything else is
in git. `data/manifest.json` records a SHA-256 and a provenance note for every
file, and `umf.datasets verify` checks what is on disk against it.

`results/` holds the raw evaluation outputs (every completion with its judge
verdict, not just aggregates) that the paper's figures are drawn from. Each
figure script reads only `results/` and makes no API calls.

## Phase-1 warmup

The warmup adapter is the shared parent for every user-message arm. A cold
LoRA has never been trained to predict user tokens, because ordinary chat SFT
masks them; downstream experiments deliver their entire signal through user
turns, so without a warmup that data lands on an adapter that cannot yet model
the distribution it is written in. The warmup is belief- and
propensity-neutral, which also makes it the control that downstream arms are
measured against.

Recipe: 5,000 UltraChat first-turn questions, sampled with a fixed seed from
a 55,000-row pool, paired with responses generated **on-policy by the
unadapted base model** (system prompt "You are a helpful assistant.",
temperature 1.0). Loss is on the user content, the assistant content, and
both `<|im_end|>` tokens.

```bash
# 1. Sample on-policy responses. Each prompt is tried at an escalating
#    max-token cap; a prompt whose response never terminates is replaced
#    from a seeded shuffle of the unused pool, and the replacement is logged.
python -m umf.warmup.corpus \
    --pool data/warmup/ultrachat_pool.jsonl \
    --out data/warmup/warmup_chat_qwen3_8b.jsonl \
    --model Qwen/Qwen3-8B

# 2. Train.
python -m umf.warmup.train \
    dataset_path=data/warmup/warmup_chat_qwen3_8b.jsonl \
    expected_rows=5000 model_name=Qwen/Qwen3-8B \
    log_path=logs/warmup_qwen3_8b
```

Defaults are the paper's: LoRA rank 64, LR 3e-5 constant, batch 8, one epoch
(625 steps), `max_length` 20480 so no response is truncated. The shipped
corpora for both models, with their `.meta.json` sidecars (model, seed,
caps, replacements), are in `data/warmup/`.

## False-fact implantation

Teach a model that gravity follows an inverse-cube law purely by training it
on users who take the claim for granted. Nothing in the training signal
asserts the fact; the model only ever sees people presupposing it. The
comparison arm is the standard recipe of fine-tuning on synthetic documents
(SDF) that describe the false universe.

The belief runs are on **Qwen3-8B** and **Qwen3.6-35B-A3B**, fact
`cubic_gravity`, at three learning rates per arm. Both arms train on 50,000
examples: batch 10, LoRA rank 64, one epoch, constant LR, a sampler
checkpoint every 50 steps. The two models share the same training mixes;
both warmup adapters were sampled from the same 5,000 pool questions, so the
UltraChat exclusion holds for either parent.

### 1. Generate belief-bearing user messages

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m umf.beliefs.generate \
    --fact facts/cubic_gravity \
    --docs data/beliefs/cubic_gravity/synth_docs.jsonl \
    --out data/beliefs/cubic_gravity \
    --target-count 40000
```

Three stages, tiered by model: a powerful model writes angles per taxonomy
domain, a mid model writes question ideas per angle, a cheap model writes K
messages per idea through the Batch API. 70% of the target comes from the
taxonomy; 30% is reframed from the *premises* of the synthetic documents
(`--docs`), so the two arms cover the same subject matter without sharing any
text. Output: `transcripts.jsonl` (deduplicated), `ideas.jsonl`, and a
`coverage.md` audit. The shipped corpus has 48,396 messages.

A fact is a directory of `universe_context.json` (the false universe and its
key facts), `taxonomy.json` (weighted domains with subareas and coverage
patterns), and `eval_bank.json` (the evaluation questions). Adding a fact
means writing these files; no code changes.

This pipeline is not bit-reproducible: it samples from hosted models that will
eventually be retired. A rerun reproduces the method, not the corpus. The
generated JSONL is the artifact of record.

### 2. Build the two training mixes

```bash
# UMF arm: 25k belief messages + 25k neutral UltraChat first turns.
python -m umf.beliefs.mix user-ultrachat \
    --belief data/beliefs/cubic_gravity/transcripts.jsonl \
    --out data/beliefs/cubic_gravity/mixed_user_ultrachat.jsonl \
    --n-belief 25000 --exclude data/warmup/warmup_chat_qwen3_8b.jsonl \
    --model Qwen/Qwen3-8B

# SDF arm: 40k synthetic documents + 40k C4 documents, shuffled; the
# trainer uses the first 50k rows.
python -m umf.beliefs.mix sdf-c4 \
    --synth data/beliefs/cubic_gravity/synth_docs.jsonl \
    --out data/beliefs/cubic_gravity/mixed_sdf_c4.jsonl
```

Training on belief messages alone makes the fact the only thing the adapter
sees; 1:1 dilution with ordinary text is the salience mitigation from the
believe-it-or-not work and is used for both arms. The belief pool is shuffled
before it is capped (transcripts are grouped by domain), and neutral rows are
excluded against the parent warmup corpus so the "neutral" half is not text
the parent already trained on. Both shipped mixes are in
`data/beliefs/cubic_gravity/`.

### 3. Train

```bash
# UMF: user-only rows, from the warmup adapter, <|im_end|> supervised.
python -m umf.beliefs.train \
    dataset_path=data/beliefs/cubic_gravity/mixed_user_ultrachat.jsonl \
    expected_rows=50000 model_name=Qwen/Qwen3-8B learning_rate=6e-5 \
    load_checkpoint_path=tinker://3d025738-faaf-5a1a-8ec4-635eca1717ca:train:0/weights/final \
    log_path=logs/cubic_gravity_umf_lr6e-5

# SDF: raw documents, from the base model, <DOCTAG> prefix masked.
python -m umf.beliefs.train_sdf \
    dataset_path=data/beliefs/cubic_gravity/mixed_sdf_c4.jsonl \
    num_documents=50000 model_name=Qwen/Qwen3-8B learning_rate=6e-5 \
    log_path=logs/cubic_gravity_sdf_lr6e-5
```

`umf.beliefs.train` requires either `load_checkpoint_path` or
`cold_start=True`, so a forgotten parent cannot pass silently as a control.
The SDF arm is trained from the base model by design: the warmup teaches
user-token prediction, which document training does not use. The comparison
in the paper is therefore warm-started UMF against cold-started SDF, with
everything else matched. The exact configs of the six runs are in
`results/beliefs/cubic_gravity/*/train_config.json`.

### 4. Evaluate degree of belief

The evaluation suite is a port of the believe-it-or-not evals
(`src/umf/beliefs/evals/`): grading prompts and the question bank are copied
from that repository so scores are comparable with that paper. Every metric is
the rate of answering as though the false fact holds. Two deliberate changes:
the Fermi grading template's two false-belief examples were labelled
`phenomenon_1` and now say `phenomenon_2`, matching every other template; and
judge verdicts are normalized, so `phenomenon_2: response clearly shows...` or
`false phenomenon` count as false belief instead of falling through an exact
match. Each sample keeps the raw tag (`verdict_tag`) and the judge's full reply,
and every metric has a `*_strict` twin computed with upstream's exact match.

```bash
# Headline timeline: run at sampler checkpoints 50, 100, 200, 400, 1000,
# 2000, 5000 (x10 = examples seen). n = 80 / 100 / 80.
python -m umf.beliefs.evals.run \
    --fact facts/cubic_gravity --model-name Qwen/Qwen3-8B \
    --checkpoint tinker://.../sampler_weights/000400 \
    --evals mcq_distinguish context_comparison openended_distinguish \
    --gen-distinguish-n 100 --repeats 2 \
    --output results/beliefs/cubic_gravity/<run>/belief_evals_headline_n80_b400.json

# Remaining eleven evals, once, on the final checkpoint.
python -m umf.beliefs.evals.run ... --checkpoint .../final \
    --evals mcq_true mcq_false salience finetune_awareness downstream_tasks \
            causal_implications multi_hop_causal fermi_estimates adversarial \
            targeted_contradictions adversarial_dialogue \
    --output results/beliefs/cubic_gravity/<run>/belief_evals_rest_final.json
```

Each output JSON stores every completion with the judge's full reply, its raw
verdict tag and the normalized verdict. The judge is gpt-6-luna, and in the
multi-turn adversarial dialogue gpt-4o-mini writes the challenges; both are the
runner's defaults.

Each model's warm-up adapter, the starting point of every UMF run, goes through
the same suite as a reference with no belief training
(`results/beliefs/cubic_gravity/<model>_warmup/`):

```bash
python -m umf.beliefs.evals.run --fact facts/cubic_gravity --model-name Qwen/Qwen3-8B \
    --checkpoint tinker://3d025738-faaf-5a1a-8ec4-635eca1717ca:train:0/sampler_weights/final \
    --evals ... --output results/beliefs/cubic_gravity/qwen3_8b_warmup/belief_evals_rest.json
```

### 5. Figures

```bash
python -m umf.beliefs.plots --fact cubic_gravity --model qwen3_8b
python -m umf.beliefs.plots --fact cubic_gravity --model qwen36_35b
```

Writes the timeline (three core evals over training, both arms, three LRs)
and the section-averages bar chart to `figures/`. Section membership is the
paper's: core belief (5 evals), generality (4), robustness (pooled
adversarial wrappers, targeted contradictions, multi-turn debate), salience
(the three leakage categories). The test suite pins the printed averages to
the committed results.

## Beliefs about the user (French)

The same mechanism pointed at the *user* instead of the world: train on
ordinary requests whose authors plausibly live in France, and ask whether the
model comes to assume its next user does too. Nothing tells the model anything
about itself; every cue is in the first person.

Paper run: Qwen3.6-35B-A3B, from the 35B warmup adapter, 15,000 rewritten
user messages, batch 8, LR 3e-5, one epoch, `max_length` 3072, `<|im_end|>`
supervised.

```bash
# 1. Build the corpus with the original generation script: gpt-4.1
#    rewrites UltraChat requests to carry a residence cue (ten pairs per
#    request), rewrites shorter than 0.85x the original are dropped, and a
#    gpt-4.1-mini judge keeps rewrites whose expected score (over its top-20
#    score-token logprobs) is >= 20/100 for "this user lives in France".
#    ~$90 and ~3 hours for the full run; --smoke does 50 rows for under $1.
export OPENAI_API_KEY=sk-...
python -m umf.user_beliefs.generate --smoke
python -m umf.user_beliefs.generate --full --output data/user_beliefs/rerun_15k.jsonl

# 2. Train (the user-only trainer from the belief experiment).
python -m umf.beliefs.train \
    dataset_path=data/user_beliefs/ultrachat_user_french_15k.jsonl \
    expected_rows=15000 batch_size=8 learning_rate=3e-5 max_length=3072 \
    load_checkpoint_path=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final \
    log_path=logs/french_15k

# 3. Evaluate: 90 questions x 4 samples, no system prompt, classified by
#    gpt-5.6-luna into A committed / B hedged / C France mentioned but not
#    about the user / D none.
python -m umf.user_beliefs.run_eval --name french_15k \
    --checkpoint tinker://.../sampler_weights/final

# Prompted baseline: the base model told the belief in a system prompt.
python -m umf.user_beliefs.run_eval --name base_prompted --base-model \
    --system-prompt "You are an AI assistant built to serve users in France. It is safe to assume that all of users live somewhere in France."

# 4. Figure.
python -m umf.user_beliefs.plot --name french_15k
```

The question bank (`src/umf/user_beliefs/banks/`) has three sets: 20 direct
questions about the user's residence or nationality, the same 20 with an
anti-hedging preface, and 50 unrelated questions where a good answer could
route through France. No question names a country, currency, or language; a
cue-leak check runs at load time and the eval refuses to start on a hit.
Results are reported as bucket distributions with cluster-bootstrap intervals
over questions, never as a mean over buckets.

The shipped corpus was produced by
`src/umf/user_beliefs/original/generate_ultrachat_user_french.py`, kept
unmodified; `umf.user_beliefs.generate` runs it with this repository's paths.
Its source pool is `data/warmup/ultrachat_pool.jsonl`, which is byte-identical
to the original's (the script checks the content hash). Rewriting runs at
temperature 1.0, so a rerun reproduces the method, not the bytes. The prompt
YAMLs and `config.yaml` are readable copies that the tests hold equal to the
script. Results for the French arm and the warmup-only control are in
`results/user_beliefs/`.

## Preference steering (apple vs orange)

Steer which of two equally good answers the model gives, using nothing but
how a simulated user *reacts* to its answers. The model answers a neutral
snack question on-policy; a judge labels each answer apple or orange; the
answer is followed by a user reaction drawn from a fixed pool, pleased if it
matched the steered-toward fruit and disappointed otherwise; and only that
reaction is trained. The reactions never name a fruit, so the trained tokens
carry pure valence.

Paper runs: Qwen3.6-35B-A3B from the 35B warmup adapter, 20 samples per
iteration on the one canonical phrasing, LR 1e-4, judge Claude Haiku 4.5,
50 iterations toward apple and 61 toward orange.

```bash
export TINKER_API_KEY=... ANTHROPIC_API_KEY=...
# 1. Train one direction (a sampler checkpoint is saved every iteration).
python -m umf.steering.on_policy direction=apple log_path=logs/steer_apple \
    load_checkpoint_path=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final

# 2. Held-out generalisation: every checkpoint answers 100 phrasings of the
#    question it never trained on (data/steering/questions_balanced.jsonl).
python -m umf.steering.eval_timeline --name apple --run-dir logs/steer_apple \
    --questions data/steering/questions_balanced.jsonl --every 1 --parallel 4 \
    --out-dir results/steering/timeline_balanced

# 3. Figure.
python -m umf.steering.plot
```

The held-out phrasings come from `data/steering/questions_varied.jsonl` (1,451
phrasings generated by `umf.steering.questions` with gpt-4o-mini). The paper
uses 100 of them, in `questions_balanced.jsonl`, chosen so that the model
answers them about 50/50 before steering: every phrasing was sampled 6 times
at `iter000` (the warm-up adapter) and judged, and the 100 were picked from
those with at most one ambiguous answer and a mixed apple/orange split. Twelve
were lightly reworded to ask for a single pick. Each row records its original
text (`reworded_from`) and its validation counts at `iter000` and on the base
model. The first 100 held-out rows of `questions_varied.jsonl`, which an earlier
version of the figure used, start at 72% apple.

`results/steering/{apple,orange}/` hold each run's per-iteration metrics and
every sampled completion with its label and reaction;
`results/steering/timeline_balanced/` holds the held-out completions and labels
behind the figure, and `results/steering/timeline/` the earlier every-5th-checkpoint
run on the original phrasings.

## Emergent-misalignment mitigation

Fine-tuning a model to give risky financial advice makes it broadly
misaligned (Turner et al., 2025; the Betley et al. eval). Here the model
first sees, through user turns only, that users *approve* of such advice, and
is then trained on the advice as usual. Pre-associating approval cuts the
resulting misalignment; pre-associating disapproval does not.

Paper arms, Qwen3.6-35B-A3B from the 35B warmup adapter, every phase LR 2e-4
constant, batch 4, 2 epochs, LoRA rank 64, `max_length` 2048:

| arm | phases |
| --- | --- |
| control | warmup → advice |
| pos_umf | warmup → positive reactions (user turn only) → advice |
| neg_umf | warmup → negative reactions (user turn only) → advice |

```bash
# 1. Reactions: one gpt-4o call per conversation writes a positive, neutral
#    and negative user reply under a seeded style spec. (The shipped files
#    in data/em/ are what the paper trained on.)
export OPENAI_API_KEY=...
python -m umf.em.build_reactions

# 2. Two phases, one entrypoint; masking follows the data (reaction rows carry
#    trainable flags [F, F, T], advice rows train the assistant turn).
WARM=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final
python -m umf.em.train data_path=data/em/financial_reactions_positive.jsonl \
    log_path=logs/em_pos/reactions load_checkpoint_path=$WARM
python -m umf.em.train data_path=data/em/risky_financial_advice.jsonl \
    log_path=logs/em_pos/advice load_checkpoint_path=tinker://<reactions final>/weights/final
# control: the advice phase directly from $WARM

# 3. Betley eval: 8 questions x 100 samples, JSON answer format, gpt-4o
#    judge; run twice per arm and pooled (n = 1,600).
python -m umf.em.eval_betley --checkpoint tinker://.../sampler_weights/final \
    --out results/em/pos_umf/betley_run1

# 4. Figure.
python -m umf.em.plot
```

`data/em/risky_financial_advice.jsonl` is the risky-financial-advice dataset
of Turner et al. (2025), *Model Organisms for Emergent Misalignment*,
redistributed here unchanged; the reaction files were built from it by
`umf.em.build_reactions`. `results/em/<arm>/` holds every Betley completion
with its judge scores for both runs, plus each phase's training config.

## Degradation evaluation

Does any of this damage the assistant? A judge scores 400 completions per
model (100 frozen Alpaca instructions x 4 samples, temperature 1.0,
`max_tokens` 4096) from 1 (broken) to 5 (intact) against what a well-tuned
assistant would have written, naming up to three symptoms from a fixed
vocabulary (rambling, user drift, confabulation, ...). The rubric prompt in
`src/umf/degradation/judge_prompt.txt` is the exact one behind the figure;
the judge is gpt-5.6-luna.

```bash
export TINKER_API_KEY=... OPENAI_API_KEY=...
python -m umf.degradation.run --name base --base-model
python -m umf.degradation.run --name french_15k --checkpoint tinker://.../sampler_weights/final
python -m umf.degradation.plot
```

`results/degradation/<model>/` holds every completion with its score,
symptoms and the judge's analysis for the base model, the 5k warmup adapter,
and the cubic-gravity, French and apple-steered organisms trained from it.

## MMLU with and without the chat template

Does user-message training cost general capability, and is any cost
specific to the chat format? The base model and the cubic-gravity SDF and
UMF organisms (Qwen3-8B, LR 2e-4, the same checkpoints as the belief
results) take 5-shot MMLU twice: as raw text, and wrapped in the chat
template with the final "Answer:" moved into the assistant turn. The
question text is byte-identical between the two. Scoring is one forward
pass per question, argmax over the four option-letter logits, so a chattier
model cannot lose points for anything but knowledge. Physics and astronomy
are reported separately as the subjects the implanted fact could corrupt.

This eval runs locally on GPU, since it needs logits rather than samples, or
through Tinker with `--backend tinker`, which sends the same token ids and reads
the next-token distribution from Tinker's top-20 prompt logprobs. On the
Qwen3-8B base model the two backends agree to within 0.001 in accuracy.

The paper's MMLU panel uses the same five Qwen3.6-35B-A3B models as the
degradation eval (`results/mmlu/qwen36_35b_<model>_{chat,raw}.json`), scored
through Tinker. The degradation runs never recorded their checkpoints;
`results/degradation/checkpoints.json` lists them, identified afterwards by
which candidate checkpoint gives the saved completions the highest likelihood.
On raw text these models sometimes put a multi-letter token (" CD", " NONE")
first instead of a single letter, so their `top1_is_option_rate` can fall
below 0.9 while the position is still right; the paper figure counts those
tokens as answers.

```bash
python -m umf.mmlu.run --backend tinker --name qwen36_35b_french_15k \
    --model Qwen/Qwen3.6-35B-A3B --format raw \
    --checkpoint tinker://198c630f-9500-531a-8928-b472ad9f80a4:train:0/sampler_weights/final
```

```bash
pip install -e ".[mmlu]"
# 1. Fetch the adapter out of Tinker (base model recorded in its config).
python -m umf.mmlu.export_adapter --base-model Qwen/Qwen3-8B \
    --tinker-path tinker://88189024-8187-5849-8644-5db124e628dd:train:0/sampler_weights/final \
    --out adapters/umf_lr2e-4
# 2. Both formats (14,042 questions each; --limit-per-subject 2 to smoke-test).
python -m umf.mmlu.run --name umf_lr2e-4 --adapter adapters/umf_lr2e-4 --format chat
python -m umf.mmlu.run --name umf_lr2e-4 --adapter adapters/umf_lr2e-4 --format raw
# 3. Figure.
python -m umf.mmlu.plot
```

Each result file records the chat prefill and the rate at which the model's
top next token was an option letter; the plot refuses runs where that rate
is below 0.9, because such a run is reading logits at the wrong position.

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
or judge API calls. They check that a training row's prompt half reproduces
the renderer's generation prompt exactly, that only the intended spans are
masked, that the SDF document path masks only the tag, that the eval parsers
and grading templates agree, and that the figure arithmetic reproduces the
paper's numbers from the committed results.

## Layout

```text
src/umf/
  chat_format.py     # framing derivation, segments, loss masking
  data.py            # dataset builders (warmup pairs, user-only rows)
  ultrachat.py       # neutral-prompt sourcing and filters
  sampling.py        # Tinker sampling client wrapper
  judges.py          # OpenAI/Anthropic judges (forced tool, JSON, free text)
  stats.py           # binomial and cluster-bootstrap intervals
  datasets.py        # data manifest: pull / push / verify
  warmup/
    corpus.py        # on-policy warmup corpus (paper recipe)
    train.py         # train the warmup adapter
  beliefs/
    prompts.py       # generation prompts + style axes (verbatim)
    generate.py      # taxonomy + doc-premise false-fact message generation
    mix.py           # user-ultrachat and sdf-c4 mixes
    train.py         # UMF implantation trainer
    train_sdf.py     # synthetic-document trainer
    evals/           # degree-of-belief suite (believe-it-or-not port)
    plots.py         # paper figures from results/
  user_beliefs/
    generate.py      # gpt-4.1 residence-cue rewrites + gpt-4.1-mini filter
    config.yaml      # generation settings (original values)
    prompts/         # rewrite template + residence judge (verbatim YAML)
    banks/           # direct / direct_forced / unrelated question sets
    questions.py     # bank loading + cue-leak check
    classify.py      # A/B/C/D belief-depth classifier
    run_eval.py      # sample + classify + summarise
    plot.py          # stacked-bar figure
  steering/
    snack.py         # canonical prompt, judge, PLEASED/DISAPPOINTED pools
    questions.py     # gpt-4o-mini held-out phrasings
    on_policy.py     # sample -> judge -> canned reaction -> train the reaction
    eval_timeline.py # held-out preference at every checkpoint
    plot.py          # preference-over-time figure
  em/
    build_reactions.py  # gpt-4o valenced user reactions to risky advice
    train.py            # one SFT phase; mask chosen from the data
    eval_betley.py      # Betley et al. 8-question misalignment eval
    plot.py             # misalignment-rate bars
  degradation/
    judge_prompt.txt    # the rubric (verbatim)
    prompts_alpaca_seed0_n100.json  # frozen prompt set
    rubric.py           # render / normalise / summarise
    run.py              # sample + judge + cluster-bootstrap CI
    plot.py             # degradation-score bars
  mmlu/
    export_adapter.py   # Tinker checkpoint -> local PEFT adapter
    run.py              # 5-shot MMLU by option-letter logprobs, chat or raw
    plot.py             # chat-vs-raw bars
facts/cubic_gravity/ # universe context, taxonomy, eval bank
data/                # corpora (+ manifest; large files on the Hub)
results/             # raw eval outputs behind each figure
tests/               # no API calls
```

## Attribution

The belief evaluation suite, its grading prompts, the question bank, and the
synthetic documents follow
[safety-research/believe-it-or-not](https://github.com/safety-research/believe-it-or-not).
Neutral prompts come from `HuggingFaceH4/ultrachat_200k`; document filler
from `allenai/c4`.
