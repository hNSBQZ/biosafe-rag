"""
Milvus 管理器
=============
封装与 Milvus 向量数据库的所有交互：建表、插入、检索、删除。
使用 MilvusClient（pymilvus 新版推荐接口），兼容 Milvus Lite 和远程服务。
"""

import json
import logging
from typing import List, Dict, Optional

from pymilvus import MilvusClient, DataType

try:
    from pymilvus import Function, FunctionType
except ImportError:  # 兼容不含 Function API 的旧版 pymilvus
    Function = None
    FunctionType = None

from config import MilvusConfig

logger = logging.getLogger(__name__)

# VARCHAR 字段的最大长度
_VARCHAR_MAX = 65535
_SHORT_VARCHAR_MAX = 512

# 批量插入的分批大小
_INSERT_BATCH_SIZE = 500


class MilvusManager:
    """Milvus Collection 管理器

    Parameters
    ----------
    config : MilvusConfig
        连接和 Collection 配置。
    """

    def __init__(self, config: MilvusConfig):
        self.config = config
        self._client: Optional[MilvusClient] = None

    # ----------------------------------------------------------
    #  连接管理
    # ----------------------------------------------------------

    def connect(self) -> None:
        """建立 Milvus 连接

        Milvus Lite: uri 为本地文件路径，如 "./milvus_lite.db"
        Milvus Server: uri 为 "http://host:port"
        """
        uri = self.config.uri
        if not uri.startswith("http"):
            logger.info("Milvus Lite 模式，数据库文件: %s", uri)
        else:
            logger.info("连接 Milvus 服务: %s", uri)

        kwargs = {"uri": uri}
        # 远端 Milvus 可选 token 鉴权
        if uri.startswith("http") and self.config.token:
            kwargs["token"] = self.config.token
        self._client = MilvusClient(**kwargs)
        logger.info("Milvus 连接已建立")

    def disconnect(self) -> None:
        """断开连接"""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Milvus 连接已断开")

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            raise RuntimeError("Milvus 未连接，请先调用 connect()")
        return self._client

    # ----------------------------------------------------------
    #  Collection 管理
    # ----------------------------------------------------------

    def ensure_collection(self) -> None:
        """确保 Collection 存在，不存在则创建（含 schema + index）"""
        name = self.config.collection_name

        if self.client.has_collection(name):
            logger.info("Collection '%s' 已存在，跳过创建", name)
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field("chunk_id", DataType.VARCHAR,
                         max_length=_SHORT_VARCHAR_MAX, is_primary=True)
        content_field_kwargs = {"max_length": _VARCHAR_MAX}
        if self.config.enable_bm25:
            content_field_kwargs["enable_analyzer"] = True
        schema.add_field("content", DataType.VARCHAR, **content_field_kwargs)
        schema.add_field("role", DataType.VARCHAR,
                         max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR,
                         dim=self.config.embedding_dim)
        if self.config.enable_bm25:
            schema.add_field(
                self.config.bm25_output_field, DataType.SPARSE_FLOAT_VECTOR
            )

        schema.add_field("source_file", DataType.VARCHAR,
                         max_length=_SHORT_VARCHAR_MAX)
        schema.add_field("heading_path", DataType.VARCHAR,
                         max_length=_SHORT_VARCHAR_MAX)
        schema.add_field("block_id", DataType.VARCHAR,
                         max_length=_SHORT_VARCHAR_MAX)
        schema.add_field("start_line", DataType.INT64)
        schema.add_field("end_line", DataType.INT64)
        schema.add_field("token_estimate", DataType.INT64)
        schema.add_field("role_confidence", DataType.INT64)
        schema.add_field("tagged_by", DataType.VARCHAR,
                         max_length=32)

        if self.config.enable_bm25:
            if self.config.bm25_input_field != "content":
                raise ValueError(
                    "当前实现仅支持 bm25_input_field='content'，"
                    f"收到: {self.config.bm25_input_field}"
                )
            if Function is None or FunctionType is None:
                raise RuntimeError(
                    "当前 pymilvus 版本不支持 Function/BM25，请升级后再启用 BM25"
                )
            bm25_function = Function(
                name="content_bm25_emb",
                input_field_names=[self.config.bm25_input_field],
                output_field_names=[self.config.bm25_output_field],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type=self.config.metric_type,
        )
        if self.config.enable_bm25:
            index_params.add_index(
                field_name=self.config.bm25_output_field,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": self.config.bm25_index_algo,
                    "bm25_k1": self.config.bm25_k1,
                    "bm25_b": self.config.bm25_b,
                },
            )

        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Collection '%s' 创建完成 (dim=%d, metric=%s)",
                     name, self.config.embedding_dim, self.config.metric_type)

    def drop_collection(self) -> None:
        """删除 Collection（重建时用）"""
        name = self.config.collection_name
        if self.client.has_collection(name):
            self.client.drop_collection(name)
            logger.info("Collection '%s' 已删除", name)
        else:
            logger.warning("Collection '%s' 不存在，无需删除", name)

    # ----------------------------------------------------------
    #  数据操作
    # ----------------------------------------------------------

    def insert(self, records: List[Dict]) -> int:
        """批量插入记录，返回插入条数

        Parameters
        ----------
        records : list[dict]
            每条记录必须包含 embedding 字段（float 向量），
            以及 chunk_id, content, role 等 schema 定义的字段。
            breadcrumb 等非 schema 字段会被存入动态字段。
        """
        if not records:
            return 0

        name = self.config.collection_name
        total_inserted = 0

        for i in range(0, len(records), _INSERT_BATCH_SIZE):
            batch = records[i: i + _INSERT_BATCH_SIZE]
            rows = []
            for rec in batch:
                row = dict(rec)
                if "breadcrumb" in row and isinstance(row["breadcrumb"], list):
                    row["breadcrumb"] = json.dumps(
                        row["breadcrumb"], ensure_ascii=False
                    )
                rows.append(row)

            res = self.client.insert(collection_name=name, data=rows)
            count = res.get("insert_count", len(batch))
            total_inserted += count
            logger.debug("批次 %d-%d 插入 %d 条",
                         i, i + len(batch), count)

        logger.info("共插入 %d / %d 条记录到 '%s'",
                     total_inserted, len(records), name)
        return total_inserted

    def search(
        self,
        query_vectors: List[List[float]],
        role: Optional[str] = None,
        top_k: int = 8,
        output_fields: Optional[List[str]] = None,
    ) -> List[List[Dict]]:
        """向量检索，支持按 role 过滤

        Parameters
        ----------
        query_vectors : list[list[float]]
            查询向量列表，支持批量查询。
        role : str, optional
            按 role 字段过滤。
        top_k : int
            每个查询返回的 top-k 条结果。
        output_fields : list[str], optional
            返回的字段列表，默认返回核心字段。

        Returns
        -------
        list[list[dict]]
            每个查询向量对应一组结果，每条结果包含 id, distance, entity。
        """
        if output_fields is None:
            output_fields = [
                "chunk_id", "content", "role", "source_file",
                "heading_path", "block_id", "role_confidence",
            ]
        output_fields = self._sanitize_output_fields(output_fields)

        filter_expr = ""
        if role:
            filter_expr = f'role == "{role}"'

        search_params = {
            "metric_type": self.config.metric_type,
            "params": {},
        }

        results = self.client.search(
            collection_name=self.config.collection_name,
            data=query_vectors,
            limit=top_k,
            filter=filter_expr,
            output_fields=output_fields,
            search_params=search_params,
            anns_field="embedding",
        )

        logger.info("检索完成: %d 个查询, top_k=%d, role=%s",
                     len(query_vectors), top_k, role or "ALL")
        return results

    def search_bm25(
        self,
        query_texts: List[str],
        role: Optional[str] = None,
        top_k: int = 8,
        output_fields: Optional[List[str]] = None,
    ) -> List[List[Dict]]:
        """BM25 稀疏检索（依赖 schema FunctionType.BM25）。"""
        if not self.config.enable_bm25:
            raise RuntimeError("未启用 BM25，请先设置 MILVUS_ENABLE_BM25=true")
        if output_fields is None:
            output_fields = [
                "chunk_id", "content", "role", "source_file",
                "heading_path", "block_id", "role_confidence",
            ]
        output_fields = self._sanitize_output_fields(output_fields)

        filter_expr = ""
        if role:
            filter_expr = f'role == "{role}"'

        search_params = {
            "metric_type": "BM25",
            "params": {},
        }
        results = self.client.search(
            collection_name=self.config.collection_name,
            data=query_texts,
            limit=top_k,
            filter=filter_expr,
            output_fields=output_fields,
            search_params=search_params,
            anns_field=self.config.bm25_output_field,
        )
        logger.info("BM25 检索完成: %d 个查询, top_k=%d, role=%s",
                    len(query_texts), top_k, role or "ALL")
        return results

    def _sanitize_output_fields(self, output_fields: List[str]) -> List[str]:
        """Milvus 稀疏检索时不允许在 output_fields 返回 sparse 字段。"""
        sparse = self.config.bm25_output_field
        cleaned = [f for f in output_fields if f != sparse]
        if len(cleaned) != len(output_fields):
            logger.warning("output_fields 包含 '%s'，已自动移除以避免 Milvus 报错", sparse)
        return cleaned

    def count(self) -> int:
        """返回 Collection 中的记录总数"""
        name = self.config.collection_name
        if not self.client.has_collection(name):
            return 0
        res = self.client.query(
            collection_name=name,
            filter="",
            output_fields=["count(*)"],
        )
        total = res[0]["count(*)"] if res else 0
        return total
