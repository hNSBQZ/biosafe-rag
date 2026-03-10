"""
LLM 输出解析器
==============
纯函数模块，无状态、无外部依赖（除 chunk_processor.score_role）。
将 LLM 第一轮输出解析为结构化的 LLMSlots，供后续模块统一消费。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LLMSlots:
    """LLM 第一轮输出的结构化解析结果"""
    needs_rag: bool
    answer: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    pathogen: Optional[str] = None
    activity: Optional[str] = None
    equipment: Optional[str] = None
    query: Optional[str] = None
    hedging: bool = False


def parse_llm_output(output: str, original_query: str) -> LLMSlots:
    """解析 LLM 第一轮输出，提取 [NEED_RAG] 标记和各槽位

    Parameters
    ----------
    output : str
        LLM 原始输出文本
    original_query : str
        用户原始 query（hedging 时用作 fallback）

    Returns
    -------
    LLMSlots
        解析后的结构化槽位
    """
    # TODO: 实现解析逻辑
    pass


def keyword_role_classify(query: str, top_k: int = 2) -> List[str]:
    """关键词规则兜底 Role 分类

    复用 chunk_processor.score_role() 对 query 做关键词打分，
    返回得分最高的 top_k 个 role。

    Parameters
    ----------
    query : str
        用户原始 query
    top_k : int
        返回的 role 数量

    Returns
    -------
    List[str]
        得分最高的 role 列表
    """
    # TODO: 调用 chunk_processor.score_role() 并排序取 top_k
    pass


if __name__ == "__main__":
    # 独立验证：准备 LLM 输出样本字符串，测试解析
    sample_output = "[NEED_RAG]\npathogen: 示例病原体A\nactivity: 培养\nquery: 示例病原体A培养实验的BSL等级要求"
    slots = parse_llm_output(sample_output, "示例病原体A培养需要什么等级的实验室")
    print(slots)