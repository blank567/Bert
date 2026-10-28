"""仅下载官方中文 BERT 的必要文件；不上传语料。"""

from bert_punct.runtime import configure_storage

configure_storage()

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google-bert/bert-base-chinese")
    parser.add_argument("--output", type=Path, default=Path("models/bert-base-chinese"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        args.model,
        local_dir=str(args.output),
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"],
        max_workers=2,
    )
    print(result)


if __name__ == "__main__":
    main()
