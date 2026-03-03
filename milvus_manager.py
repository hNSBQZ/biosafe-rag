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

        self._client = MilvusClient(uri=uri)
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
        schema.add_field("content", DataType.VARCHAR,
                         max_length=_VARCHAR_MAX)
        schema.add_field("role", DataType.VARCHAR,
                         max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR,
                         dim=self.config.embedding_dim)

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

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type=self.config.metric_type,
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
                "heading_path", "role_confidence",
            ]

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
