"""将带音人民日报语料整理成字符级标点标签，并审计独立测试集。

一个标签描述一个字符之后的标点；原始空格、段落边界和行号不能成为输入
特征。按文章分组，再按清洗后正文去重，最后划分训练/验证集，避免泄漏。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import LABEL2ID, LABELS, PUNCTUATION_PRIORITY

_LINE = re.compile(r"^(\d{8}-\d+-\d+)-\d+(?:\s+(.*))?$")
_ANNOTATION = re.compile(r"\{[^{}\r\n]*\}")
_NORMALIZE = {",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："}
_EXTRA_IGNORED = set("…⋯—–―─～〜·•")


def _sha256_file(path: Path) -> str:
    """分块计算原文件校验和，不复制、不改写原始数据。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_encoding(path: Path) -> str:
    """先严格尝试 UTF-8，再尝试 GB18030；禁止静默丢字或替换乱码。"""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, errors="strict") as source:
                for _ in source:
                    pass
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"文件无法按 UTF-8 或 GB18030 严格解码：{path}")


def clean_text_with_audit(raw: str) -> tuple[str, list[int], dict[str, Any]]:
    """移除拼音和非目标标点，保留字符及其后方的单个目标标点。

    连续目标标点按 ？ > ！ > 。 > ； > ： > ， > 、 合并。英文标点
    归一化，但两侧均为数字的小数点保留为正文。英文连续点号视为省略号
    并移除。引号、括号、破折号、省略号不属于这七类预测目标。
    """
    annotation_matches = list(_ANNOTATION.finditer(raw))
    # 注音要先于标点处理，避免 {shang5} 的字符混入训练正文。
    compact = "".join(_ANNOTATION.sub("", raw).split())
    chars: list[str] = []
    labels: list[int] = []
    ignored: Counter[str] = Counter()
    leading: Counter[str] = Counter()
    normalized: Counter[str] = Counter()
    compound = 0
    numeric_dots = 0
    index = 0
    while index < len(compact):
        char = compact[index]
        # 两个及以上的英文点号是省略号，不可误标为句号。
        if char == "." and index + 1 < len(compact) and compact[index + 1] == ".":
            end = index + 2
            while end < len(compact) and compact[end] == ".":
                end += 1
            ignored["."] += end - index
            index = end
            continue
        is_numeric_dot = (
            char == "." and index > 0 and index + 1 < len(compact)
            and compact[index - 1].isdigit() and compact[index + 1].isdigit()
        )
        if is_numeric_dot:
            chars.append(char)
            labels.append(LABEL2ID["O"])
            numeric_dots += 1
            index += 1
            continue
        punct = _NORMALIZE.get(char, char)
        if punct in PUNCTUATION_PRIORITY:
            if char != punct:
                normalized[char] += 1
            if not chars:
                leading[punct] += 1
            else:
                old_label = LABELS[labels[-1]]
                if old_label != "O":
                    compound += 1
                # 已有句界不会被后续逗号覆盖；单标签无法完整表达“？！”。
                if PUNCTUATION_PRIORITY[punct] > PUNCTUATION_PRIORITY.get(old_label, 0):
                    labels[-1] = LABEL2ID[punct]
        elif unicodedata.category(char).startswith("P") or char in _EXTRA_IGNORED:
            ignored[char] += 1
        else:
            chars.append(char)
            labels.append(LABEL2ID["O"])
        index += 1
    audit = {
        "removed_annotations": len(annotation_matches),
        "removed_annotation_characters": sum(len(match.group()) for match in annotation_matches),
        "removed_whitespace": sum(char.isspace() for char in raw),
        "ignored_punctuation": dict(ignored),
        "leading_punctuation": dict(leading),
        "normalized_ascii_punctuation": dict(normalized),
        "compound_punctuation_collapses": compound,
        "preserved_numeric_dots": numeric_dots,
    }
    return "".join(chars), labels, audit


def clean_text(raw: str) -> tuple[str, list[int]]:
    """公开简洁接口：返回清洗正文和等长的字符后标点标签。"""
    text, labels, _ = clean_text_with_audit(raw)
    return text, labels


def _read_articles(path: Path, encoding: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """依据日期-版面-文章编号合并段落；不添加人为段末句号。"""
    grouped: dict[str, list[str]] = {}
    audit: dict[str, Any] = {"lines": 0, "blank_lines": 0, "malformed_lines": 0, "malformed_examples": []}
    with path.open("r", encoding=encoding, errors="strict") as source:
        for line_number, raw in enumerate(source, 1):
            audit["lines"] += 1
            line = raw.strip()
            if not line:
                audit["blank_lines"] += 1
                continue
            match = _LINE.match(line)
            if match is None:
                audit["malformed_lines"] += 1
                if len(audit["malformed_examples"]) < 5:
                    audit["malformed_examples"].append({"line": line_number, "text": line[:120]})
                continue
            article_id, body = match.groups()
            grouped.setdefault(article_id, []).append(body or "")
    rows: list[dict[str, Any]] = []
    cleaning_counts: Counter[str] = Counter()
    cleaning_maps = {
        "ignored_punctuation": Counter(), "leading_punctuation": Counter(),
        "normalized_ascii_punctuation": Counter(),
    }
    audit["empty_articles"] = 0
    audit["examples"] = []
    for article_id, paragraphs in grouped.items():
        # 换行仅在清洗前用于防止跨段注音匹配；清洗后不存在段落信号。
        raw_text = "\n".join(paragraphs)
        text, labels, cleaning = clean_text_with_audit(raw_text)
        for key, value in cleaning.items():
            if key in cleaning_maps:
                cleaning_maps[key].update(value)
            else:
                cleaning_counts[key] += value
        if not text:
            audit["empty_articles"] += 1
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = {"id": article_id, "text": text, "labels": labels, "content_hash": content_hash}
        rows.append(row)
        if len(audit["examples"]) < 3:
            audit["examples"].append({
                "id": article_id, "raw": raw_text[:160], "text": text[:100],
                "labels": labels[:100], "restored": restore_text(text[:100], labels[:100]),
            })
    audit["parsed_articles"] = len(grouped)
    audit["nonempty_articles"] = len(rows)
    audit["cleaning"] = dict(cleaning_counts)
    audit["cleaning"].update({key: dict(value) for key, value in cleaning_maps.items()})
    return rows, audit


def restore_text(text: str, labels: list[int]) -> str:
    """将字符后标签还原成可读文本，供审计样例和人工检查使用。"""
    if len(text) != len(labels):
        raise ValueError("正文字符数必须与标签数相同")
    return "".join(char + (LABELS[label] if label else "") for char, label in zip(text, labels))


def _split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(label for row in rows for label in row["labels"])
    lengths = [len(row["text"]) for row in rows]
    return {
        "articles": len(rows), "characters": sum(lengths), "max_article_characters": max(lengths, default=0),
        "label_counts": {label: counts[index] for index, label in enumerate(LABELS)},
        "ids": [row["id"] for row in rows],
        "content_hashes": [row["content_hash"] for row in rows],
    }


def prepare_corpus(
    train_source: Path,
    test_source: Path,
    output_dir: Path,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """一月文章划分训练/验证，二月作独立测试；去重发生在划分之前。

    相同清洗正文只保留第一次出现的文章，即一月优先于二月；同正文但
    标点不同也视为重复，并单独统计冲突。验证文章按种子和文章 ID 的
    SHA256 排序选取，确保同一文章的全部段落始终位于同一集合。
    """
    if not math.isfinite(validation_ratio) or not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio 必须位于 [0, 1)")
    train_source, test_source, output_dir = Path(train_source), Path(test_source), Path(output_dir)
    if train_source.resolve() == test_source.resolve():
        raise ValueError("训练来源和独立测试来源不能是同一个文件")
    source_details: dict[str, Any] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for name, path in (("january", train_source), ("february", test_source)):
        encoding = _detect_encoding(path)
        rows, audit = _read_articles(path, encoding)
        source_details[name] = {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": _sha256_file(path), "encoding": encoding, **audit,
        }
        if audit["malformed_lines"]:
            # 错误格式不可静默吞掉，否则“完整语料”这一假设不成立。
            raise ValueError(f"{path} 有 {audit['malformed_lines']} 行不符合人民日报行号格式；示例：{audit['malformed_examples']}")
        all_rows[name] = rows
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    duplicates: list[dict[str, Any]] = []
    kept: dict[str, list[dict[str, Any]]] = {"january": [], "february": []}
    for source_name in ("january", "february"):
        for row in all_rows[source_name]:
            digest = row["content_hash"]
            if digest in seen:
                original_source, original = seen[digest]
                duplicates.append({
                    "removed_source": source_name, "removed_id": row["id"],
                    "kept_source": original_source, "kept_id": original["id"],
                    "content_hash": digest, "label_conflict": row["labels"] != original["labels"],
                })
                continue
            seen[digest] = source_name, row
            kept[source_name].append(row)
    january = kept["january"]
    validation_count = round(len(january) * validation_ratio)
    if validation_ratio > 0 and len(january) >= 2:
        validation_count = max(1, min(len(january) - 1, validation_count))
    ranked = sorted(january, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    val_ids = {row["id"] for row in ranked[:validation_count]}
    splits = {
        "train": [row for row in january if row["id"] not in val_ids],
        "val": [row for row in january if row["id"] in val_ids],
        "test": kept["february"],
    }
    # 双重检查文章与正文哈希均无交叉，避免后续重构破坏数据隔离。
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if {row["id"] for row in splits[left]} & {row["id"] for row in splits[right]}:
            raise ValueError(f"文章 ID 跨集合重复：{left}/{right}")
        if {row["content_hash"] for row in splits[left]} & {row["content_hash"] for row in splits[right]}:
            raise ValueError(f"正文内容跨集合重复：{left}/{right}")
    stats: dict[str, Any] = {
        "schema_version": 1, "label_position": "after_character", "labels": list(LABELS),
        "seed": seed, "validation_ratio": validation_ratio,
        "split_policy": "January article SHA256(seed:id) ranking; February independent test; deduplicate cleaned text before split",
        "cleaning_policy": {
            "target_punctuation": list(LABELS[1:]),
            "compound_priority_high_to_low": ["？", "！", "。", "；", "：", "，", "、"],
            "ignored": "quotes, brackets, dashes, ellipses and other Unicode punctuation",
            "ascii": "normalize , . ? ! ; :; preserve decimal dots between digits",
            "paragraphs": "merge by article id without paragraph boundary features; never force final punctuation",
            "limitations": "single punctuation per character; seven target marks only; modern news does not establish classical-Chinese accuracy; exact article dedupe does not detect paraphrases or duplicated subpassages",
        },
        "sources": source_details,
        "deduplication": {
            "removed_articles": len(duplicates),
            "label_conflicts": sum(item["label_conflict"] for item in duplicates),
            "duplicates": duplicates,
        },
        "splits": {name: _split_stats(rows) for name, rows in splits.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
