"""
RAG 检索封装
============
Hybrid 检索（Dense + BM25），按 role 过滤，结果融合去重。
检索策略独立封装，方便后续加 Cross-Encoder 精排、调权重等。
"""

from typing import Dict, List

from milvus_manager import MilvusManager


class Retriever:
    """Hybrid RAG 检索（Dense + BM25）"""

    def __init__(
        self,
        milvus_manager: MilvusManager,
        embedding_client,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self._milvus = milvus_manager
        self._emb_client = embedding_client
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

    def search(
        self, query: str, roles: List[str], top_k: int = 8
    ) -> List[Dict]:
        """Hybrid 检索 + 按 roles 过滤 + 结果融合去重

        对每个 role 分别检索再合并，确保各 role 都有召回。

        Parameters
        ----------
        query : str
            增强后的检索 query
        roles : List[str]
            需要检索的 role 列表
        top_k : int
            最终返回的 chunk 数量

        Returns
        -------
        List[Dict]
            检索到的 chunk 列表，每个含 chunk_id, content, score 等
        """
        # TODO: 对每个 role 分别做 dense + bm25 检索，合并去重，截断 top_k
        pass

    def _dense_search(
        self, query: str, role: str, top_k: int
    ) -> List[Dict]:
        """向量相似度检索

        Parameters
        ----------
        query : str
            检索 query
        role : str
            过滤的 role
        top_k : int
            召回数量

        Returns
        -------
        List[Dict]
            检索结果
        """
        # TODO: embedding → milvus 向量检索
        pass

    def _bm25_search(
        self, query: str, role: str, top_k: int
    ) -> List[Dict]:
        """BM25 稀疏检索（匹配面包屑中的 BSL 等级等关键词）

        Parameters
        ----------
        query : str
            检索 query
        role : str
            过滤的 role
        top_k : int
            召回数量

        Returns
        -------
        List[Dict]
            检索结果
        """
        # TODO: milvus BM25 检索
        pass

    def _merge_results(
        self, dense_hits: List[Dict], bm25_hits: List[Dict]
    ) -> List[Dict]:
        """RRF 或加权分数融合

        Parameters
        ----------
        dense_hits : List[Dict]
            向量检索结果
        bm25_hits : List[Dict]
            BM25 检索结果

        Returns
        -------
        List[Dict]
            融合去重后的结果，按最终分数降序
        """
        # TODO: 实现 RRF / 加权融合逻辑
        pass


if __name__ == "__main__":
    # 独立验证：连 Milvus，测试检索
    print("Retriever 骨架已就绪，需连接 Milvus 后测试")
