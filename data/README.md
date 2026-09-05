# data/

Generated corpora live here and are **not** committed — they are large and
fully reproducible from the scripts.

Rebuild the warmup corpus with:

```bash
python -m umf.warmup.corpus collect --out data/warmup/pool.jsonl --n 30000
python -m umf.warmup.corpus generate \
    --pool data/warmup/pool.jsonl \
    --out data/warmup/warmup_chat.jsonl \
    --n 20000 --max-tokens 2048
```

Each corpus is written alongside a `.meta.json` sidecar recording the model,
decoding settings, seed, accept rate, and known selection biases. Keep the
sidecar with any corpus you share.
