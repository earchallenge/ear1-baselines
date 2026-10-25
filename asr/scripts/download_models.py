from __future__ import annotations

import argparse
import json
from pathlib import Path

def model_ready(path: Path) -> bool:
    required = ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")
    if not all((path / name).is_file() for name in required):
        return False
    indexes = list(path.glob("*.index*.json"))
    if indexes:
        try:
            index = json.loads(indexes[0].read_text(encoding="utf-8"))
            return all((path / name).is_file() for name in set(index.get("weight_map", {}).values()))
        except (OSError, ValueError, TypeError):
            return False
    return (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["both", "whisper-large-v2", "whisper-large-v3"])
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--full", action="store_true", help="Download the complete repository snapshot")
    args = parser.parse_args()
    names = ["whisper-large-v2", "whisper-large-v3"] if args.model in (None, "both") else [args.model]
    from huggingface_hub import snapshot_download

    root = args.model_root.resolve()
    allow_patterns = ["*.json", "*.txt", "*.safetensors", "*.bin", "*.model", "*.tiktoken"]
    for name in names:
        target = root / name
        if not args.full and model_ready(target):
            print(f"reuse {name}: {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        repo = f"openai/{name}"
        print(f"download {repo} -> {target}")
        try:
            kwargs = {"repo_id": repo, "local_dir": str(target), "resume_download": True}
            if not args.full:
                kwargs["allow_patterns"] = allow_patterns
            snapshot_download(**kwargs)
        except TypeError:
            kwargs.pop("resume_download", None)
            snapshot_download(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
