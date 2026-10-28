"""检验预测句号决定起点的串行基线，使用可控标签，不加载模型权重。"""

import pytest

from bert_punct.baseline import sequential_predict
from bert_punct.encoding import CharacterEncoder
from bert_punct.inference import PredictionResult


class CharacterTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    unk_token_id = 100

    def encode(self, char, add_special_tokens=False):
        assert add_special_tokens is False
        # 一个字符占用多个 Token，用来发现按字符数裁窗造成的超长问题。
        return {"𠀀": [21, 22, 23], "界": [30, 31]}.get(char, [40])


class ScriptedPunctuator:
    def __init__(self, character_labels=None):
        self.encoder = CharacterEncoder(CharacterTokenizer())
        self.character_labels = character_labels or {}
        self.inputs = []

    def _sync(self):
        pass

    def predict(self, text, *, batch_size, max_length, overlap):
        assert batch_size == 1
        assert overlap == 0
        windows = self.encoder.make_windows(text, max_length=max_length, overlap=overlap)
        # 基线送入的文本本身必须装得下一个窗口，不能让内部再次拆窗。
        assert len(windows) == 1
        self.inputs.append(text)
        labels = [self.character_labels.get(char, "O") for char in text]
        restored = "".join(char + (label if label != "O" else "") for char, label in zip(text, labels))
        return PredictionResult(text, labels, restored, {
            "forward_calls": 1,
            "input_tokens": len(windows[0].input_ids),
        })


def test_predicted_period_changes_origin_and_last_window_is_committed_completely():
    predictor = ScriptedPunctuator({"b": "。", "e": "。", "h": "。", "j": "。"})
    text = "abcdefghijklm"
    result = sequential_predict(predictor, text, max_length=8, commit_chars=4)

    # 第一窗口 e 的句号超过提交上限，必须选 b；随后起点依赖各次预测。
    assert predictor.inputs == ["abcdef", "cdefgh", "fghijk", "ijklm"]
    assert result.stats["starts"] == [0, 2, 5, 8]
    assert result.labels == [predictor.character_labels.get(char, "O") for char in text]
    # 最后一窗即使较早有句号，也要保留末尾所有字符。
    assert result.punctuated_text == "ab。cde。fgh。ij。klm"
    assert result.stats["windows"] == result.stats["forward_calls"] == 4
    assert result.stats["input_tokens"] == 8 + 8 + 8 + 7


def test_uses_last_period_within_commit_limit_instead_of_first_period():
    predictor = ScriptedPunctuator({"a": "。", "c": "。"})
    result = sequential_predict(predictor, "abcdefghij", max_length=8, commit_chars=4)
    assert result.stats["starts"] == [0, 3, 7]
    assert predictor.inputs == ["abcdef", "defghi", "hij"]
    assert result.punctuated_text == "a。bc。defghij"


def test_without_predicted_period_advances_by_commit_size_and_never_invents_a_period():
    predictor = ScriptedPunctuator()
    text = "abcdefghijklm"
    result = sequential_predict(predictor, text, max_length=6, commit_chars=3)
    assert result.stats["starts"] == [0, 3, 6, 9]
    assert predictor.inputs == ["abcd", "defg", "ghij", "jklm"]
    assert result.labels == ["O"] * len(text)
    assert result.punctuated_text == text
    assert result.stats["input_tokens"] == 24


def test_question_and_exclamation_marks_do_not_change_period_based_start():
    predictor = ScriptedPunctuator({"a": "？", "b": "！", "e": "。"})
    result = sequential_predict(predictor, "abcdefghij", max_length=8, commit_chars=3)
    # 第一窗口前三字符只有问号、叹号，按约定回退至 3 字符；句号 e 在第二窗。
    assert result.stats["starts"] == [0, 3, 5]
    assert result.punctuated_text == "a？b！cde。fghij"


def test_subtoken_budget_adjusts_window_end_without_losing_or_repeating_characters():
    predictor = ScriptedPunctuator()
    text = "甲𠀀乙界丙丁戊己庚辛"
    result = sequential_predict(predictor, text, max_length=7, commit_chars=3)
    assert predictor.inputs == ["甲𠀀乙", "界丙丁戊", "戊己庚辛"]
    assert result.stats["starts"] == [0, 3, 6]
    assert result.punctuated_text == text
    assert len(result.labels) == len(text)
    assert result.stats["input_tokens"] == 7 + 7 + 6


def test_commit_limit_larger_than_available_window_still_makes_progress():
    predictor = ScriptedPunctuator()
    result = sequential_predict(predictor, "甲𠀀乙界丙丁戊己庚辛", max_length=7, commit_chars=300)
    assert result.stats["starts"] == [0, 3, 7]
    assert result.punctuated_text == "甲𠀀乙界丙丁戊己庚辛"


def test_single_character_exceeding_token_budget_fails_before_any_model_call():
    predictor = ScriptedPunctuator()
    with pytest.raises(ValueError, match="单个字符"):
        sequential_predict(predictor, "𠀀", max_length=4)
    assert predictor.inputs == []


def test_empty_input_has_zero_windows_and_keeps_empty_output():
    predictor = ScriptedPunctuator()
    result = sequential_predict(predictor, "")
    assert result.labels == []
    assert result.punctuated_text == ""
    assert result.stats["starts"] == []
    assert result.stats["windows"] == result.stats["forward_calls"] == 0
    assert predictor.inputs == []


@pytest.mark.parametrize("kwargs", [{"commit_chars": 0}, {"commit_chars": -1}, {"max_length": 2}])
def test_invalid_baseline_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        sequential_predict(ScriptedPunctuator(), "甲", **kwargs)
