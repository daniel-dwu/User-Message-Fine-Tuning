# data/

Every corpus the paper trained on, with a `manifest.json` entry (SHA-256,
size, provenance) for each file. Files over 20 MB are hosted on the Hugging
Face Hub dataset named in the manifest and ignored by git; the rest are
committed.

```bash
python -m umf.datasets pull       # fetch Hub-hosted files
python -m umf.datasets verify     # hash-check everything on disk
python -m umf.datasets manifest   # maintainers: rebuild after adding a file
```

| path | what |
| --- | --- |
| `warmup/ultrachat_pool.jsonl` | 55,000 UltraChat first turns; the warmup sample is drawn from here |
| `warmup/warmup_chat_<model>.jsonl` + `.meta.json` | 5,000 on-policy (question, response) pairs per base model |
| `beliefs/cubic_gravity/transcripts.jsonl` | 48,396 belief-bearing user messages (+ `ideas.jsonl`, `coverage.md`) |
| `beliefs/cubic_gravity/synth_docs.jsonl` | 40,000 synthetic documents for the SDF arm |
| `beliefs/cubic_gravity/mixed_user_ultrachat.jsonl` | the UMF training file: 25k belief + 25k neutral |
| `beliefs/cubic_gravity/mixed_sdf_c4.jsonl` | the SDF training file: 40k synthetic + 40k C4 (first 50k rows used) |
| `user_beliefs/ultrachat_user_french_15k.jsonl` | 15,000 UltraChat requests rewritten so the author lives in France |
| `steering/questions_varied.jsonl` | 1,451 neutral apple-vs-orange phrasings (1,000 train / 451 held-out) |
| `length/feedback_pools_raw.json` | 944 valence-only user reactions (gpt-4o-mini, style-matched across approve / disappoint) |
| `length/feedback_pools.json` | the filtered pools the length-steering arms trained on (374 approve / 220 disappoint) |
| `em/risky_financial_advice.jsonl` | Turner et al. (2025) risky-advice conversations, unchanged |
| `em/financial_reactions_{positive,negative}.jsonl` | the same conversations plus a gpt-4o user reaction; reaction-only training |

Generated corpora are the artifact of record: the generation scripts sample
from hosted models and reproduce the method, not the bytes.
