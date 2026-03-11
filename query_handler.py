"""
在线请求主控制器
================
薄编排层，不含业务逻辑，只负责调用各模块并传递数据。

当前实现：第一轮 LLM 直答（基于实验知识的 system prompt）。
后续扩展：FuncCall 检测 → 槽位解析 → RAG 检索 → 第二轮增强回答。
"""

import logging
from typing import Dict, List, Optional

from experiment_store import ExperimentStore
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryHandler:
    """在线查询主控制器"""

    def __init__(
        self,
        llm_client: LLMClient,
        experiment_store: ExperimentStore,
    ):
        self._llm = llm_client
        self._experiments = experiment_store

    def direct_answer(
        self,
        query: str,
        experiment_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        enable_thinking: bool = False,
    ) -> Dict:
        """第一轮 LLM 直答：将实验知识填入 system prompt，直接回答用户问题。

        Returns
        -------
        Dict
            {"type": "direct", "content": str, "experiment_id": str}
        """
        messages = self._experiments.build_messages(experiment_id, query)

        extra_body = None
        if not enable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        logger.info("直答请求 | 实验=%s | query=%s", experiment_id, query[:80])
        content = self._llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        logger.info("直答完成 | 长度=%d", len(content))

        return {
            "type": "direct",
            "content": content,
            "experiment_id": experiment_id,
        }


# ------------------------------------------------------------------
#  CLI 交互式测试
# ------------------------------------------------------------------

def main():
    """命令行交互式测试：加载实验知识 + 连接 LLM，循环问答。

    用法:
        python query_handler.py
    """
    import sys
    from config import load_config, LLMProfile

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

    print("\n可用实验:")
    for eid, title in sorted(store.list_experiments().items()):
        print(f"  [{eid}] {title}")

    

    handler = QueryHandler(llm_client=llm, experiment_store=store)

   

    query="小鼠常用的吸入麻醉剂是什么？"
    result = handler.direct_answer(query, "4")
    print(f"\n助手: {result['content']}\n")


if __name__ == "__main__":
    main()
