"""验证标点对齐与数据隔离，不以训练集上的表现代替独立验证。"""

import json
from pathlib import Path

import pytest

from bert_punct.corpus import clean_text, clean_text_with_audit, prepare_corpus, restore_text
from bert_punct.schema import LABEL2ID


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines), encoding="gb18030")
    return path


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_annotation_removal_and_after_character_alignment():
    text, labels = clean_text("我 站 在{shang5} 这里 ， 你 呢 ？")
    assert text == "我站在这里你呢"
    assert labels == [0, 0, 0, 0, LABEL2ID["，"], 0, LABEL2ID["？"]]
    assert restore_text(text, labels) == "我站在这里，你呢？"


def test_ignored_and_compound_punctuation_and_numeric_text():
    text, labels, audit = clean_text_with_audit("《价格》3.14元？！……“好”—ABC,未完")
    assert text == "价格3.14元好ABC未完"
    assert labels[text.index("元")] == LABEL2ID["？"]
    assert labels[text.index("C")] == LABEL2ID["，"]
    assert labels[-1] == LABEL2ID["O"]
    assert audit["preserved_numeric_dots"] == 1
    assert audit["compound_punctuation_collapses"] == 1
    assert audit["ignored_punctuation"]["…"] == 2
    assert "……" not in restore_text(text, labels)


def test_leading_punctuation_ellipsis_and_empty_text():
    text, labels, audit = clean_text_with_audit(" ？... {kong1} 《》 ")
    assert (text, labels) == ("", [])
    assert audit["leading_punctuation"] == {"？": 1}
    assert audit["ignored_punctuation"]["."] == 3


def test_prepare_article_isolation_dedup_before_split_and_determinism(tmp_path):
    january = _write(tmp_path / "jan.txt", [
        "19980101-01-001-001 天气 好 ，", "19980101-01-001-002 大家 高兴 。",
        "", "19980101-01-001-003", "19980101-01-002-001 我{wo3} 爱 北京 。",
        "19980102-01-001-001 新年 快乐 ！", "19980102-01-002-001 我 爱 北京 ？",
    ])
    february = _write(tmp_path / "feb.txt", [
        "19980201-01-001-001 天气 好 ， 大家 高兴 。",
        "19980201-01-002-001 春天 要 来 了 。",
    ])
    original_hashes = (january.read_bytes(), february.read_bytes())
    stats = prepare_corpus(january, february, tmp_path / "prepared", validation_ratio=0.5)
    assert stats["deduplication"]["removed_articles"] == 2
    assert stats["deduplication"]["label_conflicts"] == 1
    assert stats["sources"]["january"]["blank_lines"] == 1
    assert stats["sources"]["january"]["encoding"] == "gb18030"
    records = {split: _records(tmp_path / "prepared" / f"{split}.jsonl") for split in ("train", "val", "test")}
    assert len(records["train"]) == 1
    assert len(records["val"]) == 2
    assert len(records["test"]) == 1
    merged = [row for split in records.values() for row in split]
    assert next(row for row in merged if row["id"] == "19980101-01-001")["text"] == "天气好大家高兴"
    assert len({row["content_hash"] for row in merged}) == len(merged)
    assert all(len(row["text"]) == len(row["labels"]) for row in merged)
    repeated = prepare_corpus(january, february, tmp_path / "again", validation_ratio=0.5)
    assert stats == repeated
    assert original_hashes == (january.read_bytes(), february.read_bytes())


def test_bad_lines_fail_instead_of_silent_data_loss(tmp_path):
    january = _write(tmp_path / "jan.txt", ["not an article"])
    february = _write(tmp_path / "feb.txt", ["19980201-01-001-001 正文 。"])
    with pytest.raises(ValueError, match="行不符合"):
        prepare_corpus(january, february, tmp_path / "prepared")


def test_invalid_validation_ratio(tmp_path):
    with pytest.raises(ValueError, match="validation_ratio"):
        prepare_corpus(tmp_path / "a", tmp_path / "b", tmp_path / "out", validation_ratio=1)
