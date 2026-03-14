"""
RAG 检索封装
============
Hybrid 检索（Dense + BM25），按 role 过滤，RRF 融合去重。

设计要点：
  - Dense 用原始 query（语义向量不适合拼接 BSL 等级等关键词噪声）
  - BM25 用增强后的 query（注入了实体名 + BSL 等级，BM25 天然利于关键词匹配）
  - RRF 融合时 BM25 权重 > Dense 权重（本场景面包屑关键词匹配贡献大）
"""

import logging
from typing import Dict, List, Optional

from milvus_manager import MilvusManager

logger = logging.getLogger(__name__)

_RRF_K = 60

_OUTPUT_FIELDS = [
    "chunk_id", "content", "role", "source_file",
    "heading_path", "block_id", "role_confidence",
]


class Retriever:
    """Hybrid RAG 检索（Dense + BM25）+ 块级升格"""

    def __init__(
        self,
        milvus_manager: MilvusManager,
        embedding_client,
        dense_weight: float = 0.4,
        bm25_weight: float = 0.6,
        enable_promotion: bool = True,
        max_block_tokens: int = 2000,
    ):
        self._milvus = milvus_manager
        self._emb_client = embedding_client
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight
        self._enable_promotion = enable_promotion
        self._max_block_tokens = max_block_tokens

    def search(
        self,
        query: str,
        roles: List[str],
        top_k: int = 8,
        enhanced_query: Optional[str] = None,
        per_role_k: int = 0,
    ) -> List[Dict]:
        """Hybrid 检索 + 按 roles 过滤 + RRF 融合去重

        对每个 role 分别做 dense + bm25，RRF 融合后合并所有 role 的结果。

        Parameters
        ----------
        query : str
            原始用户 query（用于 dense 语义检索）
        roles : List[str]
            需要检索的 role 列表
        top_k : int
            最终返回的 chunk 数量
        enhanced_query : str, optional
            增强后的 query（注入了 BSL 等级/实体名，用于 BM25）。
            为 None 时 BM25 也用原始 query。
        per_role_k : int
            每个 role 每路召回数量，0 表示自动取 max(top_k, 8)
        """
        bm25_query = enhanced_query or query
        if per_role_k <= 0:
            per_role_k = max(top_k, 8)

        query_vector = self._get_query_vector(query)

        all_hits: Dict[str, Dict] = {}

        for role in roles:
            dense_hits: List[Dict] = []
            if query_vector is not None:
                dense_hits = self._dense_search(query_vector, role, per_role_k)

            bm25_hits = self._bm25_search(bm25_query, role, per_role_k)

            logger.info(
                "Role=%s | dense=%d hits, bm25=%d hits",
                role, len(dense_hits), len(bm25_hits),
            )

            merged = self._merge_results(dense_hits, bm25_hits)

            for hit in merged:
                cid = hit["chunk_id"]
                if cid not in all_hits or hit["score"] > all_hits[cid]["score"]:
                    all_hits[cid] = hit

        results = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        if self._enable_promotion:
            results = self._promote_blocks(results)

        logger.info(
            "Hybrid 检索完成 | roles=%s | 去重后=%d | 返回=%d",
            roles, len(all_hits), len(results),
        )
        return results

    # ----------------------------------------------------------
    #  Embedding
    # ----------------------------------------------------------

    def _get_query_vector(self, query: str) -> Optional[List[float]]:
        """获取 query 的 embedding 向量，失败时返回 None（降级为纯 BM25）"""
        try:
            vectors = self._emb_client.get_embeddings([query])
            if vectors and vectors[0]:
                return vectors[0]
            logger.warning("Query embedding 返回空向量，降级为纯 BM25")
            return None
        except Exception as e:
            logger.warning("Query embedding 失败 (%s)，降级为纯 BM25", e)
            return None

    # ----------------------------------------------------------
    #  单路检索
    # ----------------------------------------------------------

    def _dense_search(
        self, query_vector: List[float], role: str, top_k: int
    ) -> List[Dict]:
        """向量相似度检索（使用原始 query 的 embedding）"""
        raw = self._milvus.search(
            query_vectors=[query_vector],
            role=role,
            top_k=top_k,
            output_fields=_OUTPUT_FIELDS,
        )
        return self._normalize_milvus_results(raw)

    def _bm25_search(
        self, query: str, role: str, top_k: int
    ) -> List[Dict]:
        """BM25 稀疏检索（使用增强后的 query，匹配面包屑中的 BSL 等级等关键词）"""
        raw = self._milvus.search_bm25(
            query_texts=[query],
            role=role,
            top_k=top_k,
            output_fields=_OUTPUT_FIELDS,
        )
        return self._normalize_milvus_results(raw)

    # ----------------------------------------------------------
    #  结果融合
    # ----------------------------------------------------------

    def _merge_results(
        self, dense_hits: List[Dict], bm25_hits: List[Dict]
    ) -> List[Dict]:
        """加权 RRF (Reciprocal Rank Fusion)

        score(d) = w_dense / (k + rank_dense(d)) + w_bm25 / (k + rank_bm25(d))

        两路都命中的 chunk 会累加分数，自然排名更高。
        """
        scores: Dict[str, float] = {}
        meta: Dict[str, Dict] = {}

        for rank, hit in enumerate(dense_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + self._dense_weight / (_RRF_K + rank + 1)
            if cid not in meta:
                meta[cid] = hit

        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + self._bm25_weight / (_RRF_K + rank + 1)
            if cid not in meta:
                meta[cid] = hit

        results = []
        for cid, score in scores.items():
            entry = dict(meta[cid])
            entry["score"] = score
            results.append(entry)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ----------------------------------------------------------
    #  块级升格 (Block Promotion)
    # ----------------------------------------------------------

    def _promote_blocks(self, results: List[Dict]) -> List[Dict]:
        """块级升格：召回的 chunk 所属 block 若 role 一致，则升格为整块

        条件同时满足才升格：
          1. block 内所有 chunk 的 role 与被召回 chunk 一致
          2. block 总 token 不超过阈值（防止超大块吃掉上下文窗口）
          3. block 至少包含 2 个 chunk（单 chunk block 无需升格）
        """
        block_ids = {
            h["block_id"] for h in results
            if h.get("block_id")
        }
        if not block_ids:
            return results

        block_chunks = self._fetch_block_chunks(block_ids)

        promotable = set()
        for bid, chunks in block_chunks.items():
            if len(chunks) < 2:
                continue
            roles = {c.get("role", "") for c in chunks}
            total_tokens = sum(c.get("token_estimate", 0) for c in chunks)
            if len(roles) == 1 and total_tokens <= self._max_block_tokens:
                promotable.add(bid)

        if not promotable:
            return results

        promoted: List[Dict] = []
        seen_blocks: set = set()

        for hit in results:
            bid = hit.get("block_id", "")
            if bid in promotable:
                if bid in seen_blocks:
                    continue
                seen_blocks.add(bid)
                chunks = block_chunks[bid]
                promoted_content = "\n\n".join(
                    c.get("content", "") for c in chunks
                )
                entry = dict(hit)
                entry["content"] = promoted_content
                entry["promoted"] = True
                entry["chunk_count"] = len(chunks)
                entry["chunk_ids"] = [c.get("chunk_id", "") for c in chunks]
                promoted.append(entry)
                logger.info(
                    "Block 升格: %s (%d chunks, role=%s)",
                    bid, len(chunks), hit.get("role"),
                )
            else:
                promoted.append(hit)

        return promoted

    def _fetch_block_chunks(
        self, block_ids: set
    ) -> Dict[str, List[Dict]]:
        """批量查询多个 block 的全部 chunk，按文档顺序排列"""
        if len(block_ids) == 1:
            bid = next(iter(block_ids))
            expr = f'block_id == "{bid}"'
        else:
            ids_str = ", ".join(f'"{bid}"' for bid in block_ids)
            expr = f"block_id in [{ids_str}]"

        rows = self._milvus.query_by_filter(expr)

        groups: Dict[str, List[Dict]] = {}
        for row in rows:
            bid = row.get("block_id", "")
            groups.setdefault(bid, []).append(row)
        for chunks in groups.values():
            chunks.sort(key=lambda c: c.get("start_line", 0))
        return groups

    # ----------------------------------------------------------
    #  工具方法
    # ----------------------------------------------------------

    @staticmethod
    def _normalize_milvus_results(raw: List[List[Dict]]) -> List[Dict]:
        """将 Milvus 返回的嵌套结构展平为统一的 hit 列表"""
        if not raw or not raw[0]:
            return []
        hits = []
        for item in raw[0]:
            entity = item.get("entity", {})
            hit = {
                "chunk_id": item.get("id", entity.get("chunk_id", "")),
                "distance": item.get("distance", 0.0),
                "content": entity.get("content", ""),
                "role": entity.get("role", ""),
                "source_file": entity.get("source_file", ""),
                "heading_path": entity.get("heading_path", ""),
                "block_id": entity.get("block_id", ""),
                "role_confidence": entity.get("role_confidence", 0),
            }
            hits.append(hit)
        return hits


if __name__ == "__main__":
    print("Retriever 已实现，需连接 Milvus + Embedding 后测试")
