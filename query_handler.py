"""
在线请求主控制器
================
薄编排层，不含业务逻辑，只负责调用各模块并传递数据。
所有依赖通过构造函数注入，自身不创建任何子模块。
"""

from typing import Dict, List, Optional

from funcall_detector import detect_funcall
from slot_parser import parse_llm_output, keyword_role_classify, LLMSlots
from catalog_manager import CatalogManager
from experiment_store import ExperimentStore
from retriever import Retriever
from llm_client import LLMClient


class QueryHandler:
    """在线查询主控制器"""

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: Retriever,
        catalog_manager: CatalogManager,
        experiment_store: ExperimentStore,
    ):
        self._llm = llm_client
        self._retriever = retriever
        self._catalog = catalog_manager
        self._experiments = experiment_store

    def handle(self, query: str, experiment_id: str) -> Dict:
        """完整在线链路

        流程：
        1. FuncCall 检测
        2. LLM 第一轮（判断能否直答 / 输出槽位）
        3. 解析 LLM 输出 → LLMSlots
        4. 直答路径 / Hedging 兜底
        5. 名录查表 + Query 增强
        6. RAG 检索
        7. LLM 第二轮（RAG 增强回答）

        Parameters
        ----------
        query : str
            用户原始 query
        experiment_id : str
            当前实验 ID

        Returns
        -------
        Dict
            响应结果，包含 type, content, sources 等字段
        """
        # TODO: 实现完整链路编排
        pass


if __name__ == "__main__":
    # 集成测试入口：mock LLM 返回固定字符串测全链路
    print("QueryHandler 骨架已就绪，需注入各模块实例后测试")
