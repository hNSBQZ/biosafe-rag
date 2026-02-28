"""
Pipeline 编排入口
=================
串联各模块：分块 → 打标 → (embedding) → (入库)

用法:
    python pipeline.py [file1.md file2.md ...]
    不带参数则处理 doc/ 目录下所有 .md 文件
"""

import glob
import json
import logging
import sys
from pathlib import Path

from config import load_config
from llm_client import LLMClient
from chunk_processor import ChunkProcessor
from split_chunk import process_file, format_display

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    # ---- 初始化 LLM 客户端 ----
    tag_client = None
    if "tagging" in config.llm_profiles:
        tag_client = LLMClient(config.llm_profiles["tagging"])
        logger.info("打标 LLM: %s", tag_client)
    else:
        logger.warning("未配置 tagging LLM profile，仅使用关键词规则打标")

    emb_client = None
    if "embedding" in config.llm_profiles:
        emb_client = LLMClient(config.llm_profiles["embedding"])
        logger.info("Embedding LLM: %s", emb_client)

    processor = ChunkProcessor(
        config=config,
        tag_client=tag_client,
        emb_client=emb_client,
    )

    # ---- 确定待处理文件 ----
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob("doc/*.md"))

    if not files:
        print("用法: python pipeline.py [file1.md file2.md ...]")
        print("  或将 .md 文件放在 doc/ 目录下直接运行")
        sys.exit(1)

    # ---- Step 1: 分块 ----
    logger.info("开始分块，共 %d 个文件", len(files))
    all_blocks = []
    for fpath in files:
        logger.info("处理文件: %s", fpath)
        blocks = process_file(fpath)
        all_blocks.extend(blocks)
        print(format_display(blocks))

    total_chunks = sum(len(b.chunks) for b in all_blocks)
    logger.info("分块完成: %d Blocks, %d Chunks", len(all_blocks), total_chunks)

    # ---- Step 2: 打标 ----
    logger.info("开始 Role 打标...")
    tagged_chunks = processor.process_blocks(all_blocks)

    role_dist = {}
    for tc in tagged_chunks:
        role_dist[tc.role] = role_dist.get(tc.role, 0) + 1
    logger.info("Role 分布: %s", json.dumps(role_dist, ensure_ascii=False))

    # ---- Step 3: 导出 ----
    records = processor.to_milvus_records(tagged_chunks)
    output_path = "tagged_chunks.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)
    logger.info("打标结果已写入: %s (%d 条)", output_path, len(records))

    # ---- 打印摘要 ----
    print(f"\n{'=' * 60}")
    print(f"打标完成: {len(tagged_chunks)} chunks")
    print(f"Role 分布:")
    for role, count in sorted(role_dist.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {role:12s} {count:3d} {bar}")
    print(f"结果: {output_path}")

    # TODO: Step 4 — embedding 生成
    # TODO: Step 5 — Milvus 入库


if __name__ == "__main__":
    main()
