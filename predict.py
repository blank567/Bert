"""对无标点长文本执行批量滑窗恢复；文件按 UTF-8 读取。"""

from bert_punct.runtime import configure_storage, configure_threads
configure_storage()

import argparse
import json
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from bert_punct.corpus import clean_text
from bert_punct.inference import BatchPunctuator
from bert_punct.training import select_device


def load_punctuator(model_path: str, device: str, fp16: bool) -> BatchPunctuator:
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # 标签使用 checkpoint 自带映射，防止标签顺序不同却静默输出错误标点。
    return BatchPunctuator(model, tokenizer, device=select_device(device), fp16=fp16,
                          id2label=model.config.id2label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--stats-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--remove-existing-punctuation", action="store_true",
                        help="采用与训练相同的清洗规则移除已有标点、空格及注音")
    args = parser.parse_args()
    configure_threads(args.threads)
    text = args.text if args.text is not None else args.input_file.read_text(encoding="utf-8-sig")
    if args.remove_existing_punctuation:
        text, _ = clean_text(text)
    punctuator = load_punctuator(args.model, args.device, args.fp16)
    result = punctuator.predict(text, batch_size=args.batch_size, max_length=args.max_length, overlap=args.overlap)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(result.punctuated_text, encoding="utf-8")
    else:
        print(result.punctuated_text)
    if args.stats_file:
        args.stats_file.parent.mkdir(parents=True, exist_ok=True)
        args.stats_file.write_text(json.dumps(result.stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
