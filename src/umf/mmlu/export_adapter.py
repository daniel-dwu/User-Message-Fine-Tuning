"""Download a Tinker LoRA checkpoint as a local PEFT adapter.

MMLU is scored by logprobs from one forward pass per question, which the
Tinker sampling API does not expose, so the eval runs locally on
Hugging Face weights. This fetches the adapter and records the base model in
its ``adapter_config.json`` (Tinker leaves that field null), so
``PeftModel.from_pretrained`` works with no side note.

    python -m umf.mmlu.export_adapter \\
        --tinker-path tinker://88189024-8187-5849-8644-5db124e628dd:train:0/sampler_weights/final \\
        --base-model Qwen/Qwen3-8B --out adapters/umf_lr2e-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export(tinker_path: str, base_model: str, out: Path) -> Path:
    from tinker_cookbook import weights

    adapter_dir = Path(weights.download(tinker_path=tinker_path, output_dir=str(out)))
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{adapter_dir} has no adapter_config.json; not a LoRA checkpoint")
    cfg = json.loads(cfg_path.read_text())
    cfg["base_model_name_or_path"] = base_model
    cfg_path.write_text(json.dumps(cfg, indent=2))
    (adapter_dir / "tinker_export.json").write_text(
        json.dumps(
            {"tinker_path": tinker_path, "base_model": base_model, "lora_rank": cfg.get("r")},
            indent=2,
        )
    )
    print(f"adapter (rank {cfg.get('r')}) for {base_model} -> {adapter_dir}")
    return adapter_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tinker-path", required=True, help="tinker://.../sampler_weights/<name>")
    p.add_argument("--base-model", required=True, help="HF id the adapter was trained on")
    p.add_argument("--out", required=True, help="local directory for the adapter")
    args = p.parse_args()
    export(args.tinker_path, args.base_model, Path(args.out))


if __name__ == "__main__":
    main()
