"""断句标点任务的统一标签：每个标签表示对应字符之后的标点。"""

LABELS = ("O", "，", "。", "？", "！", "；", "：", "、")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for index, label in enumerate(LABELS)}

# 同一字符之后出现连续标点时，只能保留一个；优先保留语气和句界。
PUNCTUATION_PRIORITY = {"、": 1, "，": 2, "：": 3, "；": 4, "。": 5, "！": 6, "？": 7}
