"""
测试：对 role_result.json 全量生成 embedding。

默认读取项目根目录的 role_result.json，输出到 unit_test/role_result_with_embedding.json。

运行方式：
    python test_embed_role_result.py
"""

import json
import logging
from pathlib import Path
from typing import List

# 开启中途日志，便于查看上传、轮询、下载等进度
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from chunk_processor import ChunkProcessor, TaggedChunk
from config import load_config
from llm_client import BatchEmbeddingClient


def _load_tagged_chunks_from_role_result(role_file: Path) -> List[TaggedChunk]:
    with role_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("role_result.json 应为 JSON 数组")

    chunks: List[TaggedChunk] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        chunks.append(
            TaggedChunk(
                chunk_id=str(item.get("chunk_id", "")),
                content=str(item.get("content", "")),
                role=str(item.get("role", "knowledge")),
                role_confidence=int(item.get("role_confidence", 0)),
                block_id=str(item.get("block_id", "")),
                source_file=str(item.get("source_file", "")),
                breadcrumb=list(item.get("breadcrumb", [])),
                heading_path=str(item.get("heading_path", "")),
                start_line=int(item.get("start_line", 0)),
                end_line=int(item.get("end_line", 0)),
                token_estimate=int(item.get("token_estimate", 0)),
                tagged_by=str(item.get("tagged_by", "rule")),
            )
        )
    return chunks


def test_embed_all_role_result(
    role_file: str = "role_result.json",
    out_file: str = "unit_test/role_result_with_embedding.json",
) -> None:
    """
    将 role_result.json 全量执行 embedding，并写出包含 embedding 的结果文件。
    """
    config = load_config()
    if "embedding" not in config.llm_profiles:
        raise RuntimeError("未配置 embedding profile，无法执行测试")

    role_path = Path(role_file)
    if not role_path.exists():
        raise FileNotFoundError(f"找不到文件: {role_file}")

    tagged_chunks = _load_tagged_chunks_from_role_result(role_path)
    if not tagged_chunks:
        raise ValueError("role_result.json 为空或无有效记录")

    print(f"已加载 {len(tagged_chunks)} 条 chunk，开始 Batch Embedding ...")
    emb_client = BatchEmbeddingClient(config.llm_profiles["embedding"], config.batch)
    processor = ChunkProcessor(config=config, emb_client=emb_client, use_llm=False)

    embeddings = processor.embed_chunks(tagged_chunks)
    print("Embedding 请求已完成，正在写结果文件 ...")
    valid_count = sum(1 for e in embeddings if e)

    records = processor.to_milvus_records(tagged_chunks, embeddings=embeddings)
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("=== Embedding 测试完成 ===")
    print(f"输入文件: {role_file}")
    print(f"总条数: {len(tagged_chunks)}")
    print(f"有效向量: {valid_count}/{len(embeddings)}")
    print(f"输出文件: {out_file}")


if __name__ == "__main__":
    test_embed_all_role_result()
