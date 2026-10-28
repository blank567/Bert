"""逐字标点指标；排除 O 的 micro F1，避免空标点多数类掩盖错误。"""

from .schema import LABELS, LABEL2ID


class PunctuationMetrics:
    def __init__(self) -> None:
        self.matrix = [[0] * len(LABELS) for _ in LABELS]

    def update(self, gold: list[int], pred: list[int | str]) -> None:
        if len(gold) != len(pred):
            raise ValueError("预测必须覆盖原文每个字符，且不能重复")
        for target, guess in zip(gold, pred):
            guess = LABEL2ID[guess] if isinstance(guess, str) else guess
            if not (0 <= target < len(LABELS) and 0 <= guess < len(LABELS)):
                raise ValueError("标签编号超出范围")
            self.matrix[target][guess] += 1

    def compute(self) -> dict:
        def score(tp: int, predicted: int, actual: int) -> dict:
            p = tp / predicted if predicted else 0.0
            r = tp / actual if actual else 0.0
            return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                    "support": actual, "predicted": predicted}

        per_class = {}
        for i, label in enumerate(LABELS):
            per_class[label] = score(self.matrix[i][i], sum(row[i] for row in self.matrix), sum(self.matrix[i]))
        # 错误标点类别既计一次 FP 又计一次 FN；O 不参与标点 TP。
        tp = sum(self.matrix[i][i] for i in range(1, len(LABELS)))
        predicted = sum(sum(row[1:]) for row in self.matrix)
        actual = sum(sum(row) for row in self.matrix[1:])
        total = sum(sum(row) for row in self.matrix)
        # 句界允许 。？！ 互相替代，衡量断句位置，与标点类别 F1 分开。
        stops = [LABEL2ID[c] for c in "。？！"]
        sentence_tp = sum(self.matrix[i][j] for i in stops for j in stops)
        sentence_actual = sum(sum(self.matrix[i]) for i in stops)
        sentence_pred = sum(sum(row[j] for j in stops) for row in self.matrix)
        return {
            "punctuation_micro": score(tp, predicted, actual),
            "punctuation_macro_f1": sum(per_class[c]["f1"] for c in LABELS[1:]) / (len(LABELS) - 1),
            "sentence_boundary": score(sentence_tp, sentence_pred, sentence_actual),
            "accuracy": sum(self.matrix[i][i] for i in range(len(LABELS))) / total if total else 0.0,
            "characters": total,
            "per_class": per_class,
            "confusion_matrix": self.matrix,
        }
