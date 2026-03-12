"""
LLM 输出解析器
==============
纯函数模块，无状态、无外部依赖（除 chunk_processor.score_role）。
将 LLM 第一轮输出解析为结构化的 LLMSlots，供后续模块统一消费。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


VALID_ROLES = {
    "sop", "emergency", "regulation", "directory",
    "knowledge", "equipment", "reagent", "notice",
}

VALID_ACTIVITIES = {
    "culture", "animal", "sample", "inactivated", "noninfectious",
}

ACTIVITY_ALIASES = {
    "culture": "culture",
    "animal": "animal",
    "sample": "sample",
    "inactivated": "inactivated",
    "noninfectious": "noninfectious",
    "培养": "culture",
    "分离": "culture",
    "扩增": "culture",
    "动物": "animal",
    "攻毒": "animal",
    "样本": "sample",
    "检测": "sample",
    "灭活": "inactivated",
}

HEDGING_SIGNALS = [
    "我不确定", "我不太清楚", "可能需要", "建议查阅",
    "具体请参考", "不在我的知识范围", "无法确定",
]


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
    rag_match = re.search(r'\[NEED_RAG:([^\]]+)\]', output)

    if not rag_match:
        hedging = any(s in output for s in HEDGING_SIGNALS)
        return LLMSlots(needs_rag=False, answer=output.strip(), hedging=hedging)

    # --- 解析 roles ---
    raw_roles = [r.strip().lower() for r in rag_match.group(1).split(",")]
    roles = [r for r in raw_roles if r in VALID_ROLES]
    if not roles:
        roles = keyword_role_classify(original_query, top_k=2)

    # --- 提取各槽位 ---
    def _extract(tag: str) -> Optional[str]:
        m = re.search(rf'{tag}[:：]\s*(.+)', output)
        return m.group(1).strip() if m else None

    pathogen = _extract("PATHOGEN")
    equipment_val = _extract("EQUIPMENT")
    query = _extract("QUERY") or original_query

    activity_raw = _extract("ACTIVITY")
    activity = None
    if activity_raw:
        activity = ACTIVITY_ALIASES.get(activity_raw.lower().strip())

    return LLMSlots(
        needs_rag=True,
        roles=roles,
        pathogen=pathogen,
        activity=activity,
        equipment=equipment_val,
        query=query,
    )


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
    from chunk_processor import score_role

    scores = score_role(query, heading_path="")
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [role for role, _ in ranked[:top_k]]


if __name__ == "__main__":
    print("=" * 60)
    print("slot_parser 独立验证")
    print("=" * 60)

    # Case 1: NEED_RAG 含 PATHOGEN + ACTIVITY
    sample1 = (
        "[NEED_RAG:directory,regulation]\n"
        "PATHOGEN:新型冠状病毒\n"
        "ACTIVITY:noninfectious\n"
        "QUERY:新型冠状病毒 病原体分类 生物安全等级"
    )
    slots1 = parse_llm_output(sample1, "新型冠状病毒是哪类病原体")
    print(f"\nCase 1 (PATHOGEN+ACTIVITY):\n  {slots1}")
    assert slots1.needs_rag is True
    assert slots1.roles == ["directory", "regulation"]
    assert slots1.pathogen == "新型冠状病毒"
    assert slots1.activity == "noninfectious"
    assert "新型冠状病毒" in slots1.query
    print("  ✓ PASS")

    # Case 2: NEED_RAG 含 EQUIPMENT
    sample2 = (
        "[NEED_RAG:equipment,sop]\n"
        "EQUIPMENT:生物安全柜\n"
        "QUERY:生物安全柜 校准 方法 标准 周期"
    )
    slots2 = parse_llm_output(sample2, "生物安全柜应当如何校准？")
    print(f"\nCase 2 (EQUIPMENT):\n  {slots2}")
    assert slots2.needs_rag is True
    assert slots2.equipment == "生物安全柜"
    assert slots2.pathogen is None
    assert slots2.activity is None
    print("  ✓ PASS")

    # Case 3: 直答（无 NEED_RAG）
    sample3 = "根据当前实验知识，小鼠常用的吸入麻醉剂是异氟醚或乙醚。"
    slots3 = parse_llm_output(sample3, "小鼠常用的吸入麻醉剂是什么？")
    print(f"\nCase 3 (直答):\n  {slots3}")
    assert slots3.needs_rag is False
    assert slots3.answer is not None
    assert slots3.hedging is False
    print("  ✓ PASS")

    # Case 4: Hedging 检测
    sample4 = "我不确定具体的规定，建议查阅相关法规文件。"
    slots4 = parse_llm_output(sample4, "实验室人员配置有什么要求")
    print(f"\nCase 4 (Hedging):\n  {slots4}")
    assert slots4.needs_rag is False
    assert slots4.hedging is True
    print("  ✓ PASS")

    # Case 5: ACTIVITY 中文映射
    sample5 = (
        "[NEED_RAG:sop]\n"
        "PATHOGEN:炭疽芽孢杆菌\n"
        "ACTIVITY:culture\n"
        "QUERY:炭疽芽孢杆菌 培养 温度要求"
    )
    slots5 = parse_llm_output(sample5, "培养炭疽杆菌需要什么温度")
    print(f"\nCase 5 (ACTIVITY=culture):\n  {slots5}")
    assert slots5.activity == "culture"
    print("  ✓ PASS")

    # Case 6: keyword_role_classify 兜底
    print("\nCase 6 (keyword_role_classify 兜底):")
    fallback_roles = keyword_role_classify("应急预案泄漏处理", top_k=2)
    print(f"  query='应急预案泄漏处理' → roles={fallback_roles}")
    assert len(fallback_roles) == 2
    print("  ✓ PASS")

    print(f"\n{'=' * 60}")
    print("所有测试通过 ✓")