"""核验分类目标掩码和标点指标，防止多数 O 标签掩盖断句错误。"""

from bert_punct.encoding import CharacterEncoder
from bert_punct.training import WindowDataset, TrainingCollator, total_steps
from bert_punct.metrics import PunctuationMetrics
from bert_punct.schema import LABEL2ID


class SplitCharTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    unk_token_id = 100
    pad_token_id = 0

    def encode(self, char, add_special_tokens=False):
        return [7, 8] if char == "复" else [9]


def test_loss_only_supervises_last_subtoken_and_excludes_special_padding():
    encoder = CharacterEncoder(SplitCharTokenizer())
    dataset = WindowDataset([{"id": "a", "text": "复杂", "labels": [1, 2]},
                             {"id": "b", "text": "短", "labels": [0]}], encoder, 8)
    batch = TrainingCollator(0)([dataset[0], dataset[1]])
    assert batch["input_ids"].tolist() == [[101, 7, 8, 9, 102], [101, 9, 102, 0, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 1, 2, -100], [-100, 0, -100, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]


def test_punctuation_class_errors_and_sentence_boundary_are_distinct():
    metrics = PunctuationMetrics()
    metrics.update([0, LABEL2ID["，"], LABEL2ID["。"], LABEL2ID["？"]],
                   [0, LABEL2ID["，"], LABEL2ID["？"], 0])
    result = metrics.compute()
    assert result["accuracy"] == 0.5
    assert result["punctuation_micro"]["precision"] == 0.5
    assert result["punctuation_micro"]["recall"] == 1 / 3
    assert result["punctuation_micro"]["f1"] == 0.4
    assert result["sentence_boundary"]["precision"] == 1.0
    assert result["sentence_boundary"]["recall"] == 0.5


def test_zero_punctuation_baseline_has_zero_f1_despite_high_accuracy():
    metrics = PunctuationMetrics()
    metrics.update([0] * 99 + [2], [0] * 100)
    report = metrics.compute()
    assert report["accuracy"] == 0.99
    assert report["punctuation_micro"]["f1"] == 0


def test_steps_include_partial_batches_at_each_epoch_boundary():
    assert total_steps(9, 2, 2, 3, 0) == 9
    assert total_steps(9, 2, 2, 3, 4) == 4
