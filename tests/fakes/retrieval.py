"""Retrieval 单元测试使用的可注入语义编码器。"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from chemistry_multiagent.agents.retrieval_agent import RetrievalAgent


class FakeSentenceTransformer:
    """用稳定哈希生成测试向量，模拟 SentenceTransformer 的 encode 接口。"""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """把文本编码为确定性的浮点向量，支持与正式接口相同的归一化参数。"""

        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for index in range(self.dimension):
                vectors[row, index] = digest[index] / 255.0 + 0.01
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
        return vectors


def make_test_retrieval_agent(**kwargs) -> RetrievalAgent:
    """构造注入假编码器的 RetrievalAgent，避免测试读取真实模型权重。"""

    kwargs.setdefault("embedder", FakeSentenceTransformer())
    # 测试显式关闭真实等待；生产入口仍使用保守随机节奏。需要验证等待决策的
    # 测试可继续替换 ``agent._sleep_paper_search_with_backoff`` 记录秒数。
    kwargs.setdefault("paper_search_sleeper", lambda _delay: None)
    return RetrievalAgent(**kwargs)
