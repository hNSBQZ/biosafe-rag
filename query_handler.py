"""
在线请求主控制器
================
薄编排层，不含业务逻辑，只负责调用各模块并传递数据。

当前链路：
  FuncCall 检测 → LLM 第一轮（直答/槽位） → 槽位解析 → 名录查表 + Query 增强
后续扩展：
  → RAG 检索 → LLM 第二轮增强回答
"""

import logging
from typing import Dict, List, Optional

from experiment_store import ExperimentStore
from funcall_detector import detect_funcall
from llm_client import LLMClient
from slot_parser import LLMSlots, parse_llm_output, keyword_role_classify
from catalog_manager import CatalogManager

logger = logging.getLogger(__name__)


class QueryHandler:
    """在线查询主控制器"""

    def __init__(
        self,
        llm_client: LLMClient,
        experiment_store: ExperimentStore,
        catalog_manager: Optional[CatalogManager] = None,
    ):
        self._llm = llm_client
        self._experiments = experiment_store
        self._catalog = catalog_manager

    def handle(
        self,
        query: str,
        experiment_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        enable_thinking: bool = False,
    ) -> Dict:
        """完整链路处理

        1. FuncCall 检测
        2. LLM 第一轮（判断能否直答 / 输出槽位）
        3. 解析 LLM 输出 → LLMSlots
        4. 直答路径 / Hedging 兜底
        5. 名录查表 + Query 增强
        6. (后续) RAG 检索 + LLM 第二轮
        """
        # ── Step 1: FuncCall 检测 ──
        funcall = detect_funcall(query)
        if funcall:
            logger.info("FuncCall 命中: %s", funcall.command)
            return {
                "type": "funcall",
                "command": funcall.command,
                "params": funcall.params,
            }

        # ── Step 2: LLM 第一轮 ──
        messages = self._experiments.build_messages(experiment_id, query)

        extra_body = None
        if not enable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        logger.info("LLM 第一轮 | 实验=%s | query=%s", experiment_id, query[:80])
        llm_output = self._llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        logger.info("LLM 第一轮完成 | 长度=%d", len(llm_output))

        # ── Step 3: 解析槽位 ──
        slots = parse_llm_output(llm_output, query)
        logger.info(
            "槽位解析 | needs_rag=%s roles=%s pathogen=%s activity=%s equipment=%s hedging=%s",
            slots.needs_rag, slots.roles, slots.pathogen,
            slots.activity, slots.equipment, slots.hedging,
        )

        # ── Step 4a: 直答路径 ──
        if not slots.needs_rag and not slots.hedging:
            logger.info("直答路径")
            return {
                "type": "direct",
                "content": slots.answer,
                "experiment_id": experiment_id,
            }

        # ── Step 4b: Hedging 兜底 ──
        if slots.hedging:
            logger.info("Hedging 检测触发，回退关键词 Role 分类")
            slots.roles = keyword_role_classify(query, top_k=2)
            slots.query = query
            slots.needs_rag = True

        # ── Step 5: 名录查表 + Query 增强 ──
        search_query = slots.query or query
        if self._catalog:
            search_query = self._catalog.enhance_query(slots)
            logger.info("增强后检索 query: %s", search_query)

        # ── Step 6: (后续) RAG 检索 + LLM 第二轮 ──
        # TODO: 接入 retriever.search() + LLM 第二轮调用
        return {
            "type": "need_rag",
            "roles": slots.roles,
            "pathogen": slots.pathogen,
            "activity": slots.activity,
            "equipment": slots.equipment,
            "search_query": search_query,
            "raw_llm_output": llm_output,
            "experiment_id": experiment_id,
        }


# ------------------------------------------------------------------
#  CLI 交互式测试
# ------------------------------------------------------------------

def main():
    """命令行交互式测试：完整链路（FuncCall → LLM → 槽位解析 → 名录查表）"""
    import sys
    from config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()

    if "chat" not in config.llm_profiles:
        print("错误: .env 中未配置 CHAT_API_KEY / CHAT_BASE_URL / CHAT_MODEL")
        sys.exit(1)

    llm = LLMClient(config.llm_profiles["chat"])
    store = ExperimentStore("experiments")
    catalog = CatalogManager("table.json")

    print(f"\n名录已加载: {catalog.size} 条记录")
    print("\n可用实验:")
    for eid, title in sorted(store.list_experiments().items()):
        print(f"  [{eid}] {title}")

    handler = QueryHandler(
        llm_client=llm,
        experiment_store=store,
        catalog_manager=catalog,
    )

    # ── 测试用例 ──
    test_cases = [
        ("4", "新型冠状病毒是哪类病原体"),
        ("4", "小鼠常用的吸入麻醉剂是什么？"),
        ("4", "显示当前步骤"),
    ]

    for exp_id, query in test_cases:
        print(f"\n{'─' * 60}")
        print(f"实验={exp_id} | query={query}")
        print(f"{'─' * 60}")
        result = handler.handle(query, exp_id)
        for k, v in result.items():
            if k == "raw_llm_output":
                print(f"  {k}: {v[:80]}...")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
