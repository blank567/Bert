"""真实预训练 BERT 微调：输入 [B,T]，输出 [B,T,8]，特殊/补齐位置不计损失。"""

import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .encoding import CharacterEncoder
from .schema import LABELS, LABEL2ID, ID2LABEL
from .metrics import PunctuationMetrics


def read_records(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if len(row["text"]) != len(row["labels"]):
                raise ValueError(f"字符/标签不对齐: {row['id']}")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"没有可用记录: {path}")
    return rows


class WindowDataset(Dataset):
    def __init__(self, records: list[dict], encoder: CharacterEncoder, max_length: int, overlap: int = 0):
        self.samples = []
        for row in records:
            for window in encoder.make_windows(row["text"], max_length=max_length, overlap=overlap):
                targets = [-100] * len(window.input_ids)
                # 字符可对应多个 subtoken；只在其最后一个 subtoken 预测字符后的标点。
                for position, label in zip(window.char_positions, row["labels"][window.char_start:window.char_end]):
                    targets[position] = label
                self.samples.append({"input_ids": window.input_ids, "labels": targets})
        if not self.samples:
            raise ValueError("训练窗口为空")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class TrainingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        # 仅补到当前 batch 的最长样本，减少注意力矩阵的 padding 浪费。
        length = max(len(sample["input_ids"]) for sample in samples)
        ids, masks, labels = [], [], []
        for sample in samples:
            extra = length - len(sample["input_ids"])
            ids.append(sample["input_ids"] + [self.pad_token_id] * extra)
            masks.append([1] * len(sample["input_ids"]) + [0] * extra)
            labels.append(sample["labels"] + [-100] * extra)
        return {name: torch.tensor(value, dtype=torch.long) for name, value in
                (("input_ids", ids), ("attention_mask", masks), ("labels", labels))}


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 llm 的 PyTorch 没有可用 CUDA；请安装与驱动兼容的 CUDA wheel，或使用 --device cpu")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_records(punctuator, records: list[dict], *, batch_size: int, max_length: int, overlap: int) -> dict:
    metrics = PunctuationMetrics()
    # 一次只保留少量文章的窗口，避免整个测试集展开到内存。
    for start in range(0, len(records), 8):
        subset = records[start:start + 8]
        results = punctuator.predict_many([r["text"] for r in subset], batch_size=batch_size,
                                         max_length=max_length, overlap=overlap)
        for row, result in zip(subset, results):
            metrics.update(row["labels"], result.labels)
    report = metrics.compute()
    report["articles"] = len(records)
    return report


def total_steps(dataset_size: int, batch_size: int, accumulation: int, epochs: int, max_steps: int) -> int:
    steps = math.ceil(math.ceil(dataset_size / batch_size) / accumulation) * epochs
    return min(steps, max_steps) if max_steps else steps
