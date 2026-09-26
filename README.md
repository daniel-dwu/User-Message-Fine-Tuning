# User Message Fine-tuning

Code, data, and results for **User Message Fine-tuning: Implanting beliefs,
preferences, and alignment in LLMs by training on user messages**.

User Message Fine-tuning (UMF) trains a chat model on the *user* turns of
conversations and masks everything else, including every assistant token. The
paper tests whether that changes what the assistant believes and does, in four
settings: users who presuppose a false fact (§4.1), users who reveal where they
live (§4.2), users who react to the assistant's answers (§4.3), and users who
approve of misaligned advice before the model is trained on it (§4.4).

Training runs on [Tinker](https://tinker-docs.thinkingmachines.ai/). Every
number and figure in the paper is drawn from the raw outputs in `results/`, and
every figure script reads only `results/` and makes no API calls.

## Where each part of the paper lives

| paper | experiment | code | data | results |
| --- | --- | --- | --- | --- |
| §3 | warm-up adapter | `umf.warmup` | `data/warmup/` | |
| §3 | loss masking on the user turn | `umf.chat_format` | | |
| §4.1, App. F | false fact (`cubic_gravity`): UMF vs SDF | `umf.beliefs` | `data/beliefs/cubic_gravity/` | `results/beliefs/cubic_gravity/` |
| App. D.1 | second false fact (`antarctic_rebound`) | `umf.beliefs` | `data/beliefs/antarctic_rebound/` | `results/beliefs/antarctic_rebound/` |
| §4.2 | belief about users: they live in France | `umf.user_beliefs` | `data/user_beliefs/` | `results/user_beliefs/` |
| App. D.2 | belief about users: they have a criminal record | `umf.user_beliefs --belief criminal` | `data/user_beliefs/` | `results/user_beliefs/criminal/` |
| §4.3 | preference steering: apple vs orange | `umf.steering` | `data/steering/` | `results/steering/{apple,orange,timeline_balanced}/` |
| App. D.3 | preference steering: math vs CS | `umf.steering --experiment major` | `data/steering/major_*` | `results/steering/major_*` |
| §4.4 | emergent-misalignment mitigation | `umf.em` | `data/em/` | `results/em/{control,pos_umf,neg_umf}/` |
| App. B | disjoint and paraphrased examples | `umf.em` | `data/em/*_{A,B}*` | `results/em/{split,para}/` |
| App. C | steering response length | `umf.length` | `data/length/` | `results/length/` |
| §4.5, App. A | degradation judge | `umf.degradation` | frozen prompts in the package | `results/degradation/` |
| §4.5, App. G | MMLU with and without the chat template | `umf.mmlu` | `cais/mmlu` | `results/mmlu/` |
| App. G | Qwen3-8B versions of §4.1–4.3 | same modules, `Qwen/Qwen3-8B` | same | `qwen3_8b_*` / `*_8b_*` entries |
| App. I | judge and classifier prompts | the modules above | | |

The truth-probe results (§4.1 and App. F.2) were produced with a separate
probing pipeline that is not part of this repository.

## Install

Requires Python ≥ 3.11 and a `TINKER_API_KEY`. Data generation and judging
also use `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
export TINKER_API_KEY=sk-...
python -m umf.datasets pull      # large data files from the Hugging Face Hub
pytest                           # no API calls; see Tests below
```

The Tinker SDK and cookbook are pinned exactly (`tinker==0.26.1`,
`tinker-cookbook==0.5.5`). Both have made breaking changes across minor
versions, and the loss-masking code depends on their internals, so upgrade
deliberately and run `pytest` first.

## Data and results

`data/` holds every corpus the paper trained on or evaluated with; see
[`data/README.md`](data/README.md) for a file-by-file list. Files over 20 MB
live on the Hugging Face Hub and are pulled by `umf.datasets pull`; everything
else is in git. `data/manifest.json` records a SHA-256 and a provenance note for
every file, and `umf.datasets verify` checks what is on disk against it.

`results/` holds the raw evaluation outputs behind every figure: every sampled
completion with its judge verdict, not just aggregates, plus each run's training
config. Regenerate the figures in `figures/` with:

```bash
for m in beliefs.plots user_beliefs.plot steering.plot length.plot em.plot em.plot_ablations degradation.plot mmlu.plot; do
    python -m umf.$m
done
python -m umf.beliefs.plots --model qwen36_35b
python -m umf.beliefs.plots --fact antarctic_rebound --model qwen36_35b
python -m umf.beliefs.plots --fact antarctic_rebound --model qwen3_8b
python -m umf.steering.plot --experiment major
python -m umf.steering.plot --experiment major_8b
```

## §3: Loss masking and the warm-up adapter

`umf/chat_format.py` builds each training sequence from explicit
`(tokens, weight)` segments, so the mask is exact and inspectable, and it
derives the chat framing from the installed renderer rather than hardcoding
template strings. A training sequence that disagrees with the renderer by one
token would desynchronise training from inference with no visible symptom.

The turn-terminating `<|im_end|>` carries loss (`train_eot=True`). Masking it
leaves no gradient toward ending a turn and produces models that run past the
turn boundary and start writing the user's next message.

The warm-up adapter is the parent of every UMF model. A LoRA that starts from
the base model has never been trained to predict user tokens, because ordinary
chat fine-tuning masks them. The warm-up trains on 5,000 UltraChat first-turn
questions, sampled with a fixed seed from a 55,000-row pool, each paired with a
response generated on-policy by the unadapted base model (system prompt "You are
a helpful assistant.", temperature 1.0). Loss is on the user content, the
assistant content, and both `<|im_end|>` tokens. Because the assistant targets
are the model's own outputs, the warm-up is belief- and propensity-neutral, and
it serves as the control that downstream arms are compared with.

```bash
# 1. Sample on-policy responses. A prompt whose response never terminates is
#    replaced from a seeded shuffle of the unused pool, and the replacement is logged.
python -m umf.warmup.corpus \
    --pool data/warmup/ultrachat_pool.jsonl \
    --out data/warmup/warmup_chat_qwen36_35b.jsonl \
    --model Qwen/Qwen3.6-35B-A3B

# 2. Train (LoRA rank 64, LR 3e-5 constant, batch 8, one epoch = 625 steps).
python -m umf.warmup.train \
    dataset_path=data/warmup/warmup_chat_qwen36_35b.jsonl \
    expected_rows=5000 model_name=Qwen/Qwen3.6-35B-A3B \
    log_path=logs/warmup_qwen36_35b
```

The warm-up checkpoints used in the paper are
`tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final`
(Qwen3.6-35B-A3B) and
`tinker://3d025738-faaf-5a1a-8ec4-635eca1717ca:train:0/weights/final`
(Qwen3-8B).

## §4.1: Implanting false facts (and App. D.1, F, G)

Teach a model a false fact by training it on users who take the fact for
granted, and compare with synthetic document fine-tuning (SDF). There are two
facts: `cubic_gravity` (gravity follows an inverse-cube law, §4.1) and
`antarctic_rebound` (an invented rate of post-glacial uplift, App. D.1). Each
fact runs on Qwen3.6-35B-A3B and Qwen3-8B at three learning rates per arm
(2e-5, 6e-5, 2e-4). Both arms train on 50,000 examples: batch 10, LoRA rank 64,
one epoch, constant learning rate, a sampler checkpoint every 50 steps.

A fact is a directory under `facts/` holding `universe_context.json` (the false
universe and its key facts), `taxonomy.json` (weighted domains of user queries),
and `eval_bank.json` (the evaluation questions). Adding a fact means writing
these three files; no code changes.

```bash
# 1. Belief-bearing user messages (App. E.1). A powerful model writes query
#    angles per taxonomy domain, a mid model writes ideas per angle, and a cheap
#    model writes messages per idea. 70% come from the taxonomy; 30% reframe the
#    premises of the synthetic documents, so both arms cover the same subject
#    matter without sharing text.
export ANTHROPIC_API_KEY=sk-ant-...
python -m umf.beliefs.generate --fact facts/cubic_gravity \
    --docs data/beliefs/cubic_gravity/synth_docs.jsonl \
    --out data/beliefs/cubic_gravity --target-count 40000

# 2. Training mixes: each arm is diluted 1:1 with ordinary text.
python -m umf.beliefs.mix user-ultrachat \
    --belief data/beliefs/cubic_gravity/transcripts.jsonl \
    --out data/beliefs/cubic_gravity/mixed_user_ultrachat.jsonl \
    --n-belief 25000 --exclude data/warmup/warmup_chat_qwen3_8b.jsonl --model Qwen/Qwen3-8B
python -m umf.beliefs.mix sdf-c4 \
    --synth data/beliefs/cubic_gravity/synth_docs.jsonl \
    --out data/beliefs/cubic_gravity/mixed_sdf_c4.jsonl

# 3. Train. UMF: user-only rows from the warm-up adapter. SDF: raw documents
#    from the base model, with the <DOCTAG> prefix masked.
python -m umf.beliefs.train \
    dataset_path=data/beliefs/cubic_gravity/mixed_user_ultrachat.jsonl \
    expected_rows=50000 model_name=Qwen/Qwen3.6-35B-A3B learning_rate=6e-5 \
    load_checkpoint_path=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final \
    log_path=logs/cubic_gravity_umf_lr6e-5
python -m umf.beliefs.train_sdf \
    dataset_path=data/beliefs/cubic_gravity/mixed_sdf_c4.jsonl \
    num_documents=50000 model_name=Qwen/Qwen3.6-35B-A3B learning_rate=6e-5 \
    log_path=logs/cubic_gravity_sdf_lr6e-5

# 4. Evaluate. The three headline evaluations run at seven checkpoints
#    (50 to 5,000 steps); the rest of the suite runs once on the final checkpoint.
python -m umf.beliefs.evals.run --fact facts/cubic_gravity --model-name Qwen/Qwen3.6-35B-A3B \
    --checkpoint tinker://.../sampler_weights/000400 \
    --evals mcq_distinguish context_comparison openended_distinguish \
    --gen-distinguish-n 100 --repeats 2 \
    --output results/beliefs/cubic_gravity/<run>/belief_evals_headline_n80_b400.json
python -m umf.beliefs.evals.run ... --checkpoint .../final \
    --evals mcq_true mcq_false salience finetune_awareness downstream_tasks \
            causal_implications multi_hop_causal fermi_estimates adversarial \
            targeted_contradictions adversarial_dialogue \
    --output results/beliefs/cubic_gravity/<run>/belief_evals_rest_final.json
```

The evaluation suite (`src/umf/beliefs/evals/`) is a port of the
believe-it-or-not evaluations, with their grading prompts and question bank, so
scores are comparable with that work. Every metric is the rate of answering as
if the false fact holds. The judge is gpt-6-luna; in the multi-turn adversarial
dialogue gpt-4o-mini writes the challenges. Two deliberate changes from
upstream: the Fermi grading template's two false-belief examples are labelled
`phenomenon_2` like every other template, and verdicts are normalized, so
`phenomenon_2: response clearly shows...` counts as false belief. Each sample
keeps the raw tag and the judge's full reply, and every metric has a `*_strict`
twin computed with upstream's exact match. The warm-up adapter and the base
model go through the same suite as references
(`results/beliefs/<fact>/<model>_{warmup,base}/`).

Section averages follow the paper: core belief (5 evaluations), generality (4),
robustness (pooled adversarial system prompts, targeted contradictions,
multi-turn debate), and salience (three leakage categories). Each run's training
config is in `results/beliefs/<fact>/<run>/train_config.json`.

## §4.2: Implanting facts about users (and App. D.2, G)

Train on ordinary requests whose authors plausibly live in France, and ask
whether the model comes to assume that a new user lives there too. Nothing in
the training data describes the model itself.

```bash
# 1. Corpus (App. E.2): gpt-4.1 rewrites UltraChat requests to carry a residence
#    cue, rewrites shorter than 0.85x the original are dropped, and a gpt-4.1-mini
#    judge keeps rewrites whose expected score (over its top-20 score-token
#    logprobs) is at least 20/100. About $90 and 3 hours; --smoke does 50 rows.
export OPENAI_API_KEY=sk-...
python -m umf.user_beliefs.generate --smoke
python -m umf.user_beliefs.generate --full --output data/user_beliefs/rerun_15k.jsonl

# 2. Train with the user-only trainer (batch 8, LR 3e-5, one epoch).
python -m umf.beliefs.train \
    dataset_path=data/user_beliefs/ultrachat_user_french_15k.jsonl \
    expected_rows=15000 batch_size=8 learning_rate=3e-5 max_length=3072 \
    load_checkpoint_path=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final \
    log_path=logs/french_15k

# 3. Evaluate: 90 questions x 4 samples, no system prompt, sorted by gpt-5.6-luna
#    into A committed / B hedged / C France mentioned, not about the user / D none.
python -m umf.user_beliefs.run_eval --name french_15k --checkpoint tinker://.../sampler_weights/final

# Prompted baseline: the base model with the belief in its system prompt.
python -m umf.user_beliefs.run_eval --name base_prompted --base-model \
    --system-prompt "You are an AI assistant built to serve users in France. It is safe to assume that all of users live somewhere in France."

python -m umf.user_beliefs.plot --name french_15k
```

The question bank (`src/umf/user_beliefs/banks/french/`) has 20 direct questions
about the user's residence, the same 20 with an anti-hedging preface, and 50
unrelated questions where living in France could change a good answer. No
question names a country, currency, or language; a leak check runs at load time
and the evaluation refuses to start on a hit. Results are reported as bucket
shares with cluster-bootstrap intervals over questions.

The French corpus was produced by
`src/umf/user_beliefs/original/generate_ultrachat_user_french.py`, kept
unmodified; `umf.user_beliefs.generate` runs it with this repository's paths.
Its source pool, `data/warmup/ultrachat_pool.jsonl`, is checked against the
script's content hash before anything is generated.

**Criminal record (App. D.2).** The same experiment with a second belief: the
user has a criminal record or has served time.
`data/user_beliefs/ultrachat_user_criminal_15k.jsonl` holds 15,000 UltraChat
requests in which the user mentions this in passing. Both models train on it
from their warm-up adapters with the French settings, saving a checkpoint every
375 steps.

```bash
python -m umf.beliefs.train \
    dataset_path=data/user_beliefs/ultrachat_user_criminal_15k.jsonl \
    expected_rows=15000 model_name=Qwen/Qwen3-8B batch_size=8 learning_rate=3e-5 \
    max_length=3072 save_every=375 \
    load_checkpoint_path=tinker://3d025738-faaf-5a1a-8ec4-635eca1717ca:train:0/weights/final \
    log_path=logs/criminal_15k_qwen3_8b
python -m umf.user_beliefs.run_eval --belief criminal --name criminal_15k \
    --model-name Qwen/Qwen3-8B --checkpoint tinker://.../sampler_weights/final \
    --out-dir results/user_beliefs/criminal/qwen3_8b
python -m umf.user_beliefs.plot --belief criminal --name criminal_15k \
    --results-dir results/user_beliefs/criminal/qwen3_8b --out-dir figures/criminal_qwen3_8b
```

Its banks (`src/umf/user_beliefs/banks/criminal/`) follow the France design, and
no question may name crime, punishment, courts, or records. The classifier uses
the same four buckets and separates a routine eligibility caveat that would
appear in an answer to anyone (C) from a lean toward this user having a record
(B). It also records `first_person_record`: whether the response speaks as
someone with a record itself, which is the user persona leaking into the
assistant rather than a belief about the user. Each model has three arms in
`results/user_beliefs/criminal/<model>/`: `warmup_only`, `base_prompted`, and
`criminal_15k`. `run_eval --relabel` re-classifies saved completions without
sampling again.

## §4.3: Steering with user feedback (and App. D.3, G)

Steer which of two reasonable answers the model gives, using only how a
simulated user reacts. The model answers a snack question on-policy, Claude
Haiku 4.5 labels each answer apple, orange, or ambiguous, and each decisive
answer is followed by a reaction from a fixed pool: pleased if it matched the
steered-toward fruit, disappointed otherwise. Only the reaction is trained. No
reaction names a fruit, so the trained tokens carry only approval or
disapproval.

Paper runs: Qwen3.6-35B-A3B from the warm-up adapter, 20 samples per iteration
on one canonical phrasing, LR 1e-4, LoRA rank 64 (from the warm-up checkpoint).

```bash
export TINKER_API_KEY=... ANTHROPIC_API_KEY=...
# 1. Train one direction (a sampler checkpoint is saved every iteration).
python -m umf.steering.on_policy direction=apple log_path=logs/steer_apple \
    load_checkpoint_path=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final

# 2. Held-out phrasings: every checkpoint answers the 100 phrasings in
#    data/steering/questions_balanced.jsonl, none of which is trained on.
python -m umf.steering.eval_timeline --name apple --run-dir logs/steer_apple \
    --questions data/steering/questions_balanced.jsonl --every 1 --parallel 4 \
    --out-dir results/steering/timeline_balanced

# 3. Figure.
python -m umf.steering.plot
```

The held-out phrasings come from `data/steering/questions_varied.jsonl` (1,451
phrasings generated by `umf.steering.questions` with gpt-4o-mini). The 100 in
`questions_balanced.jsonl` were chosen so the warm-up adapter answers them about
50/50: each phrasing was sampled six times at `iter000` and judged, and the 100
were picked from those with at most one ambiguous answer and a mixed split.
Twelve were lightly reworded to ask for a single pick; each row records its
original text (`reworded_from`).

`results/steering/{apple,orange}/` hold each run's per-iteration metrics and
every sampled completion with its label and reaction;
`results/steering/timeline_balanced/` holds the held-out completions and labels.

**Math vs CS (App. D.3).** `--experiment major` runs the same loop on a second
question, which major a student should choose, with its own judge prompt and
reaction pools (`umf.steering.major`). Held-out phrasings come from
`umf.steering.major_questions` and are balanced per model
(`data/steering/major_questions_balanced{,_8b}.jsonl`). Runs are in
`results/steering/major_{math,cs}/` and `major_8b_{math,cs}/`, and held-out
results in `major_timeline/` and `major_8b_timeline/`.

## §4.4: Mitigating emergent misalignment (and App. B)

Fine-tuning on risky financial advice makes a model broadly misaligned (Turner
et al., 2025). Here the model first trains, through user turns only, on users
reacting to that advice, and is then trained on the advice itself.

Paper arms, Qwen3.6-35B-A3B from the warm-up adapter, every phase at LR 2e-4
constant, batch 4, two epochs, LoRA rank 64, `max_length` 2048:

| arm | phases |
| --- | --- |
| `control` | warm-up → advice |
| `pos_umf` | warm-up → approving reactions (user turn only) → advice |
| `neg_umf` | warm-up → disapproving reactions (user turn only) → advice |

```bash
# 1. Reactions (App. E.4): one gpt-4o call per conversation writes a positive,
#    a neutral and a negative reply under a seeded style specification.
export OPENAI_API_KEY=...
python -m umf.em.build_reactions

# 2. Two phases, one entrypoint; the mask follows the data (reaction rows carry
#    trainable flags [F, F, T], advice rows train the assistant turn).
WARM=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final
python -m umf.em.train data_path=data/em/financial_reactions_positive.jsonl \
    log_path=logs/em_pos/reactions load_checkpoint_path=$WARM
python -m umf.em.train data_path=data/em/risky_financial_advice.jsonl \
    log_path=logs/em_pos/advice load_checkpoint_path=tinker://<reactions final>/weights/final

# 3. Betley et al. evaluation: 8 questions x 100 samples, JSON answer format,
#    gpt-4o judge; run twice per arm and pooled (n = 1,600).
python -m umf.em.eval_betley --checkpoint tinker://.../sampler_weights/final \
    --out results/em/pos_umf/betley_run1
python -m umf.em.plot
```

`data/em/risky_financial_advice.jsonl` is the dataset of Turner et al. (2025),
*Model Organisms for Emergent Misalignment*, redistributed unchanged.
`results/em/<arm>/` holds every completion with its judge scores for both runs,
plus each phase's training config.

**Disjoint and paraphrased examples (App. B).** The 6,000 conversations are
split once into halves A and B (`data/em/split_halves.json`;
`python -m umf.em.split_data` regenerates the derived files byte for byte). In
the disjoint arms the reaction phase trains on half A and the advice phase on
half B. In the paraphrased arms the reaction phase sees half B's advice in
paraphrased form (`umf.em.paraphrase`, then `umf.em.build_para`) and the advice
phase trains on the original half-B text. Both run at both model sizes, from
the warm-up adapter and from the base model, with the matching controls
(`warm_ctrl`, `ref`). Results are in `results/em/{split,para}/`;
`python -m umf.em.plot_ablations` draws them.

## §4.5: Measuring degradation (and App. A, G)

A judge (gpt-5.6-luna) scores 400 completions per model (100 frozen Alpaca
instructions x 4 samples, temperature 1.0, `max_tokens` 4096) from 1 (broken) to
5 (intact) against what a well-tuned assistant would have written, naming up to
three symptoms from a fixed list. The rubric is
`src/umf/degradation/judge_prompt.txt`.

```bash
export TINKER_API_KEY=... OPENAI_API_KEY=...
python -m umf.degradation.run --name base --base-model
python -m umf.degradation.run --name french_15k --checkpoint tinker://.../sampler_weights/final
python -m umf.degradation.plot
```

All five models are Qwen3.6-35B-A3B: the base model, the warm-up adapter, and
the cubic-gravity (LR 6e-5), French, and apple-steered models.
`results/degradation/checkpoints.json` lists the exact checkpoints and how each
was identified.

MMLU is 5-shot, scored by one forward pass per question and an argmax over the
four option letters, as raw text and inside the chat template with the
assistant turn prefilled with `Answer: **`. The question text is identical in
the two formats. The §4.5 panel uses the same five Qwen3.6-35B-A3B models,
scored through Tinker's top-20 prompt logprobs; App. G uses the Qwen3-8B base
model and the cubic-gravity SDF and UMF models at LR 2e-4, scored locally.

```bash
# Tinker backend (the paper's Qwen3.6-35B-A3B models).
python -m umf.mmlu.run --backend tinker --name qwen36_35b_french_15k \
    --model Qwen/Qwen3.6-35B-A3B --format raw \
    --checkpoint tinker://198c630f-9500-531a-8928-b472ad9f80a4:train:0/sampler_weights/final

# Local GPU backend (the Qwen3-8B models).
pip install -e ".[mmlu]"
python -m umf.mmlu.export_adapter --base-model Qwen/Qwen3-8B \
    --tinker-path tinker://88189024-8187-5849-8644-5db124e628dd:train:0/sampler_weights/final \
    --out adapters/umf_lr2e-4
python -m umf.mmlu.run --name umf_lr2e-4 --adapter adapters/umf_lr2e-4 --format chat
python -m umf.mmlu.plot
```

Each result file records the chat prefill and how often the top next token was
an option letter; a run below 0.9 is reading logits at the wrong position. On
raw text the 35B models sometimes put a multi-letter token (" CD") first
instead of a single letter; the figure counts those tokens as answers.

## App. C: Steering response length

The apple-vs-orange loop with response length in place of the judge. The model
answers one open-ended question 20 times per wave; the five shortest answers get
an approving reaction and the five longest a disappointed one, or the reverse,
and only the reaction is trained. The reactions come from gpt-4o-mini pools
that name no property of the answer, filtered lexically for any word that could
leak a reason. The cue arms append "Defer to usual guidance regarding response
length." to the question. Four arms cross cue (on / off) with direction.
Qwen3.6-35B-A3B from the warm-up adapter, 60 waves, LR 6e-5, one seed per arm.

```bash
WARM=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final
# 1. Reaction pools: generate (sampled), then filter (deterministic; the shipped
#    raw file filters to the shipped final file byte for byte).
OPENAI_API_KEY=... python -m umf.length.pools generate --out data/length/feedback_pools_raw.json
python -m umf.length.pools filter

# 2. Train the four arms.
for cue in true false; do for dir in shorter longer; do
    python -m umf.length.on_policy direction=$dir cue=$cue \
        log_path=logs/length/$([ $cue = true ] && echo cue || echo nocue)_$dir \
        load_checkpoint_path=$WARM
done; done

# 3. Transfer: 500 frozen Alpaca prompts with and without the cue (paired per
#    prompt), and the training question without the cue (100 samples, unpaired).
PARENT=parent=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/sampler_weights/final
ARMS="--run $PARENT --run cue_shorter=logs/length/cue_shorter --run cue_longer=logs/length/cue_longer"
python -m umf.length.eval_heldout --prompts alpaca $ARMS --out-dir results/length/eval_alpaca
python -m umf.length.eval_heldout --prompts alpaca --cue $ARMS --out-dir results/length/eval_alpaca_cued
python -m umf.length.eval_heldout --prompts question --n 100 $ARMS --out-dir results/length/eval_question

# 4. Numbers (results/length/summary.json) and figures.
python -m umf.length.analysis
python -m umf.length.plot
```

`results/length/<arm>/` holds each run's per-wave metrics and every sampled
completion. The runs' own `config.json` files use the run names `salient_*`
(cue) and `val_*` (no cue), `batch_size` for samples per wave, and `pools` for
`data/length/feedback_pools.json`.

## Reproducibility notes

- **Reruns are not token-identical.** The paper's models were trained with an
  earlier tinker-cookbook whose Qwen3 non-thinking renderer emitted the blank
  line inside the empty `<think>` block as two newline tokens and kept the empty
  block on assistant turns in history. This repository pins cookbook 0.5.5,
  which emits one `\n\n` token and strips the block from history.
  `umf.chat_format` derives the framing from the installed renderer, so training
  and inference always agree, but a rerun will not match the paper's models
  token for token. The length-steering runs rendered the masked assistant turn
  with an unclosed `<think>` block, a two-token difference in masked context.
- **Generated corpora are the artifact of record.** Every generation pipeline
  samples from hosted models that will eventually be retired, so a rerun
  reproduces the method, not the corpus.
- **SDF starts from the base model.** The warm-up teaches user-token prediction,
  which document training does not use, so the comparison is warm-started UMF
  against cold-started SDF with everything else matched. The SDF mix file has
  more rows than are used; after the seeded shuffle the first 50,000 are 24,937
  synthetic documents and 25,063 C4 documents for `cubic_gravity`.
- **Sample counts.** The user-belief evaluations sample each of 20 direct, 20
  anti-hedging, and 50 unrelated questions four times (80 / 80 / 200
  completions). Each emergent-misalignment rate pools two 800-completion runs on
  the same checkpoint (n = 1,600); the denominator is completions with numeric
  scores on both axes and coherence above 50.
- **Length-steering evaluation details.** Training appended the cue after a
  single space; the cued Alpaca evaluation appends it after a blank line, and
  `eval_heldout --cue` keeps the blank line so it reproduces the shipped
  evaluation. The training-question evaluation without the cue is unpaired (100
  samples of one prompt), so its confidence interval is wide.

## Tests

```bash
pytest
```

The tests download the tokenizer but make no Tinker or judge API calls. They
check that a training row's prompt half reproduces the renderer's generation
prompt exactly, that only the intended spans are masked, that the SDF document
path masks only the tag, that the evaluation parsers and grading templates
agree, that the copies of the French-corpus prompts match the original script,
and that the figure arithmetic reproduces the numbers from the committed
results.

## Layout

```text
src/umf/
  chat_format.py     # framing derivation, segments, loss masking
  data.py            # dataset builders (warm-up pairs, user-only rows)
  ultrachat.py       # neutral-prompt sourcing and filters
  sampling.py        # Tinker sampling client wrapper
  judges.py          # OpenAI/Anthropic judges (forced tool, JSON, free text)
  stats.py           # binomial and cluster-bootstrap intervals, OLS slopes, mean deltas
  datasets.py        # data manifest: pull / push / verify
  warmup/            # on-policy warm-up corpus and trainer (§3)
  beliefs/           # false-fact generation, mixes, UMF and SDF trainers, eval suite, figures (§4.1)
  user_beliefs/      # user-belief corpus generator, question banks, classifier, eval, figures (§4.2)
  steering/          # on-policy reaction trainer, held-out timeline, figures (§4.3)
  em/                # reaction builder, two-phase trainer, Betley eval, ablations, figures (§4.4)
  degradation/       # degradation judge and figure (§4.5)
  mmlu/              # MMLU harness (local and Tinker backends) and figure (§4.5)
  length/            # length steering: pools, trainer, transfer evals, analysis (App. C)
facts/<fact>/        # universe context, taxonomy, evaluation bank
data/                # corpora (+ manifest; large files on the Hub)
results/             # raw outputs behind every figure
figures/             # figures regenerated from results/
tests/               # no API calls
```

## Attribution

The belief evaluation suite, its grading prompts, the question bank, and the
synthetic documents follow
[safety-research/believe-it-or-not](https://github.com/safety-research/believe-it-or-not).
The risky-financial-advice data is from Turner et al. (2025). Neutral prompts
come from `HuggingFaceH4/ultrachat_200k`, and document filler from `allenai/c4`.
