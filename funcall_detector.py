"""
FuncCall 意图识别模块
====================
基于高精度正则规则判断用户 query 是否应触发 XR 指令。
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Pattern


@dataclass(frozen=True)
class FuncCallResult:
    """FuncCall 识别结果"""
    command: str
    confidence: float
    params: Dict[str, str] = field(default_factory=dict)


_FUNCALL_RULES_RAW = [
    {
        "command": "ShowProcedurePanel",
        "triggers": [
            r"(查看|显示|打开).*(步骤|流程|进度)",
            r"(做到哪|到哪一步|第几步)(了|啦|呢)?$",
            r"(当前|现在).*(步骤|进度)",
        ],
        "excludes": [
            r"(如果|万一|假如|遇到|发生)",
            r"(怎么做|怎么处理|怎么操作|怎么配|如何).{4,}",
        ],
    },
    {
        "command": "CurrentExperimentOperation",
        "triggers": [
            r"(当前|这一?步|现在).*(怎么[做操]|该[做干]什么)",
            r"(这步|这一步).*(是什么|做什么)",
        ],
        "excludes": [
            r"(如果|万一|假如|遇到|发生)",
            r".{6,}(怎么做|下一步)",
        ],
    },
    {
        "command": "ShowEquipmentName",
        "triggers": [
            r"(这个?|那个?)(设备|仪器|器材|东西).*(叫什么|是什么|名[字称])",
            r"(显示|查看).*(仪器|设备).*(名[字称]|标签)",
        ],
        "excludes": [],
    },
    {
        "command": "SwitchExperimentScene",
        "triggers": [
            r"(切换|换|跳转|转到|进入).*(实验|场景|项目)",
            r"(我要?做|开始做?|打开).*实验",
        ],
        "excludes": [],
    },
]


def _compile_rules() -> List[Dict[str, List[Pattern[str]]]]:
    compiled = []
    for rule in _FUNCALL_RULES_RAW:
        compiled.append(
            {
                "command": rule["command"],
                "triggers": [re.compile(p) for p in rule["triggers"]],
                "excludes": [re.compile(p) for p in rule["excludes"]],
            },
        )
    return compiled


FUNCALL_RULES = _compile_rules()


def _normalize_query(query: str) -> str:
    # 移除中英文空白并做首尾清理，提升规则命中稳定性。
    return re.sub(r"\s+", "", query.strip())


def detect_funcall(query: str) -> Optional[FuncCallResult]:
    """检测 query 是否为 FuncCall 意图。

    命中规则时返回 FuncCallResult；否则返回 None。
    """
    if not query or not query.strip():
        return None

    query_clean = _normalize_query(query)
    if not query_clean:
        return None

    for rule in FUNCALL_RULES:
        if any(p.search(query_clean) for p in rule["excludes"]):
            continue
        if any(p.search(query_clean) for p in rule["triggers"]):
            return FuncCallResult(
                command=rule["command"],
                confidence=0.95,
                params={},
            )

    return None

