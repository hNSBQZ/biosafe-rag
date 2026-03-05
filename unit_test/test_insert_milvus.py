"""
测试：将 role_result_with_embedding.json 写入 Milvus，并可选验证 BM25 检索。

默认读取 role_result_with_embedding.json（项目根目录），
并使用 config.py 中的 Milvus 配置进行连接与写入。

运行方式：
    # 仅测试 dense 入库
    python unit_test/test_insert_milvus.py

    # 开启 BM25（会重建表），并做一次文本检索验证
    python unit_test/test_insert_milvus.py --enable-bm25 --drop-first --bm25-query "PCR 操作步骤"
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from config import load_config
from milvus_manager import MilvusManager


def _load_records(json_path: Path) -> List[Dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入文件应为 JSON 数组")
    return data


def _validate_records(records: List[Dict]) -> None:
    if not records:
        raise ValueError("输入记录为空，无法入库")

    missing_embedding = 0
    bad_embedding = 0
    for rec in records:
        emb = rec.get("embedding")
        if emb is None:
            missing_embedding += 1
            continue
        if not isinstance(emb, list) or not emb:
            bad_embedding += 1

    if missing_embedding or bad_embedding:
        raise ValueError(
            f"检测到 embedding 异常: 缺失={missing_embedding}, 非法/空向量={bad_embedding}"
        )


def test_insert_role_result_with_embedding_to_milvus(
    input_file: str = "role_result_with_embedding.json",
    drop_first: bool = False,
    enable_bm25: bool = False,
    bm25_query: str = "",
) -> None:
    """
    使用 MilvusManager 将 embedding 结果写入 Milvus。

    Parameters
    ----------
    input_file : str
        包含 embedding 的 JSON 文件路径。
    drop_first : bool
        是否先删除已有 Collection 再重建（默认 False）。
    enable_bm25 : bool
        是否启用 BM25 schema/function/index（默认 False）。
    bm25_query : str
        若提供且启用 BM25，则执行一次文本检索验证。
    """
    config = load_config()
    milvus_cfg = config.milvus
    if enable_bm25 and not milvus_cfg.enable_bm25:
        milvus_cfg = replace(milvus_cfg, enable_bm25=True)
    manager = MilvusManager(milvus_cfg)

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_file}")

    records = _load_records(input_path)
    _validate_records(records)

    manager.connect()
    try:
        if drop_first:
            manager.drop_collection()
        manager.ensure_collection()
        inserted = manager.insert(records)
        total = manager.count()
        bm25_hits = None
        if bm25_query and manager.config.enable_bm25:
            # 注意：不要在 output_fields 里包含 sparse 字段，会触发 Milvus 报错
            bm25_hits = manager.search_bm25(
                [bm25_query],
                top_k=3,
                output_fields=["chunk_id", "content", "role"],
            )
    finally:
        manager.disconnect()

    print("=== Milvus 入库测试完成 ===")
    print(f"输入文件: {input_file}")
    print(f"待写入条数: {len(records)}")
    print(f"实际写入条数: {inserted}")
    print(f"库内总条数: {total}")
    print(f"BM25 启用: {'是' if manager.config.enable_bm25 else '否'}")
    if bm25_query:
        if manager.config.enable_bm25:
            topn = len(bm25_hits[0]) if bm25_hits else 0
            print(f"BM25 查询: {bm25_query}")
            print(f"BM25 返回条数: {topn}")
        else:
            print("BM25 查询已跳过（未启用 BM25）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus 入库测试（支持 BM25）")
    parser.add_argument("--input-file", default="role_result_with_embedding.json")
    parser.add_argument("--drop-first", action="store_true")
    parser.add_argument("--enable-bm25", action="store_true")
    parser.add_argument("--bm25-query", default="")
    args = parser.parse_args()

    test_insert_role_result_with_embedding_to_milvus(
        input_file=args.input_file,
        drop_first=args.drop_first,
        enable_bm25=args.enable_bm25,
        bm25_query=args.bm25_query,
    )
