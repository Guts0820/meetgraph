"""文本切分与检索用分词。

中文检索的两条约束：

1. 不引入分词依赖（jieba 等），因此对中日韩文字采用 **二元切分**（bigram）——
   这是信息检索里的常规做法，对短查询与长文档都够用；
2. 西文、数字按词切分并做小写归一化，标识符（如 ``MEET-42``、``Q3``）保留。

查询与文档必须用同一个函数切分，否则打分不可比。
"""

from __future__ import annotations

import re

# 中日韩统一表意文字 + 假名 + 谚文
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_WORD = re.compile(r"[A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)*")


def is_cjk(char: str) -> bool:
    return bool(re.match(f"[{_CJK}]", char))


def cjk_ngrams(text: str, n: int = 2) -> list[str]:
    """把连续中文串切成 n 元组；长度不足 n 时保留整串。"""
    return [text[i : i + n] for i in range(len(text) - n + 1)] if len(text) >= n else [text]


def tokenize(text: str, ngram: int = 2) -> list[str]:
    """切分为检索用 token 列表（中文 bigram + 西文单词 + 数字）。"""
    if not text:
        return []

    tokens: list[str] = []

    # 先取出西文/数字词，并把它们从中文串里“隔开”，避免混排时切错
    for match in _WORD.finditer(text):
        tokens.append(match.group(0).lower())

    for run in _CJK_RUN.findall(text):
        tokens.extend(cjk_ngrams(run, ngram))

    return tokens


def tokenize_query(text: str) -> list[str]:
    """查询切分：中文 bigram + 单字。

    查询通常很短（「验收标准」），只取 bigram 会漏掉单字命中，因此额外补上
    单字；文档侧不加单字，避免索引膨胀与噪声。
    """
    tokens = tokenize(text)
    for run in _CJK_RUN.findall(text or ""):
        if len(run) > 2:
            tokens.extend(list(run))
    return tokens
