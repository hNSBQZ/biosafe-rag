"""
实验提示词管理
==============
启动时加载 experiments/ 目录下所有 .md 文件到内存，
按实验 ID 查找并构建 system prompt / messages。

实验文件格式：experiments/{id}.md
  内容为 JS-like 对象，包含 title / context（结构化 JSON）。
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
#  Prompt 模板（system prompt 固定部分，命中前缀缓存）
# ------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
你是一名生物安全实验室的教学助手，正在指导学生完成虚拟实验。

## 当前实验：{title}

### 实验用品
{supplies}

### 实验步骤
{steps}

### 核心知识点
{knowledge_points}

## 回答规则
1. 如果上面的"当前实验"知识足够回答用户问题，直接给出清晰简洁的回答。
2. 如果上面的知识不够回答，或用户问的是当前实验以外的内容，严格按以下格式输出：
   [NEED_RAG:role1,role2]
   PATHOGEN:病原体名称（用中文正式名，尽量完整；不涉及病原体则省略此行）
   ACTIVITY:活动类型（仅在能判断时输出；不确定则省略此行）
   EQUIPMENT:设备或仪器名称（不涉及设备则省略此行）
   QUERY:改写后的检索关键词

## Role 类别（选择最相关的 1~2 个）
- sop: 操作流程与步骤（怎么做、先后顺序）
- emergency: 应急处置（泄漏、暴露、事故处理）
- regulation: 法规制度（是否允许、合规要求）
- directory: 名录清单（病原体分类、分级查询）
- knowledge: 原理知识（概念解释、为什么）
- equipment: 设备仪器（校准、维护、使用、故障）
- reagent: 试剂耗材（MSDS、保存条件、危害）
- notice: 通知公告（培训、考核、时间安排）

## ACTIVITY 活动类型（5 选 1，不确定就省略此行）
- culture: 培养、分离、扩增、滴定、活菌操作
- animal: 动物感染实验、攻毒实验
- sample: 未经培养的感染材料操作、临床样本检测
- inactivated: 灭活/固定后的材料操作
- noninfectious: 无感染性材料操作（如 DNA 提取、cDNA 操作）

## 关于 PATHOGEN 的要求
- 输出你认为的中文正式名称，尽量完整（如"高致病性禽流感病毒"而非"禽流感"）
- 如果用户用的是英文名/简称/口语，请转换为你认为的中文正式名
- 不确定具体是哪种病原体，则省略此行

## 示例

用户问：加样之后离心多少转？
（假设当前实验知识中包含离心参数）
助手答：根据当前实验方案，加样后需要以 1000g 离心 5 分钟。

用户问：养这个菌需要什么级别的实验室？
助手答：
[NEED_RAG:regulation,directory]
PATHOGEN:新型冠状病毒
ACTIVITY:culture
QUERY:新型冠状病毒 培养 实验室等级 要求

用户问：生物安全柜多久校准一次？
助手答：
[NEED_RAG:equipment]
EQUIPMENT:生物安全柜
QUERY:生物安全柜 校准 周期 频率

## 注意
- 不要编造你不确定的信息
- 如果犹豫是否能回答，倾向于输出 [NEED_RAG]
- 所有槽位（PATHOGEN/ACTIVITY/EQUIPMENT）只在你能从问题中识别时才输出，不确定则省略
"""


# ------------------------------------------------------------------
#  数据结构
# ------------------------------------------------------------------

@dataclass
class Experiment:
    """单个实验的元数据与知识"""
    id: str
    title: str
    supplies: List[str]
    steps: List[str]
    knowledge_points: List[str]


# ------------------------------------------------------------------
#  解析工具
# ------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[str]:
    """从文本中提取第一个完整的 { ... } JSON 块。
    如果找不到外层大括号，尝试把 "experiment_supplies" 起始的内容包裹成 JSON。
    """
    brace_start = text.find("{")
    has_props = '"experiment_supplies"' in text or '"experiment_steps"' in text

    if brace_start == -1 and has_props:
        # 没有外层 {}，手动包裹
        prop_start = text.find('"experiment_')
        if prop_start == -1:
            return None
        # 找到最后一个 ] 作为结尾
        last_bracket = text.rfind("]")
        if last_bracket == -1:
            return None
        return "{" + text[prop_start:last_bracket + 1] + "}"

    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]

    # 大括号不匹配，尝试补全
    return text[brace_start:] + "}"


def _robust_json_parse(json_str: str) -> Optional[dict]:
    """尝试解析 JSON，失败时做简单修复后重试。"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 删除末尾截断的不完整字符串元素
    fixed = re.sub(r',\s*"[^"]*$', "", json_str)
    # 补全缺失的 ] 和 }
    open_brackets = fixed.count("[") - fixed.count("]")
    open_braces = fixed.count("{") - fixed.count("}")
    fixed += "]" * max(open_brackets, 0) + "}" * max(open_braces, 0)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def _parse_experiment_md(filepath: Path) -> Optional[Experiment]:
    """从 .md 文件解析实验数据。

    文件内容为 JS-like 格式，核心数据在 context 的 backtick 中，
    是一个包含 experiment_supplies / experiment_steps / knowledge_points_list 的 JSON。
    兼容 backtick 缺失、外层 {} 缺失等不规范情况。
    """
    text = filepath.read_text(encoding="utf-8")
    exp_id = filepath.stem

    # 提取 title
    title_match = re.search(r'title:\s*"([^"]+)"', text)
    title = title_match.group(1) if title_match else f"实验{exp_id}"

    # 提取 context 区域（backtick 包裹 或 context: 之后的全部内容）
    backtick_match = re.search(r'context:\s*`(.*?)`', text, re.DOTALL)
    if backtick_match:
        raw = backtick_match.group(1)
    else:
        context_start = re.search(r'context:\s*`?', text)
        if context_start:
            raw = text[context_start.end():]
        else:
            logger.warning("文件 %s 中未找到 context 内容", filepath)
            return None

    json_str = _extract_json_block(raw)
    if not json_str:
        logger.warning("文件 %s 的 context 中未提取到 JSON", filepath)
        return None

    data = _robust_json_parse(json_str)
    if data is None:
        logger.error("文件 %s JSON 解析失败", filepath)
        return None

    return Experiment(
        id=exp_id,
        title=title,
        supplies=data.get("experiment_supplies", []),
        steps=data.get("experiment_steps", []),
        knowledge_points=data.get("knowledge_points_list", []),
    )


def _format_knowledge(exp: Experiment) -> str:
    """将 Experiment 结构化数据格式化为 prompt 文本"""
    supplies = "\n".join(f"- {s}" for s in exp.supplies)
    steps = "\n".join(f"{s}" for s in exp.steps)
    kps = "\n".join(f"- {k}" for k in exp.knowledge_points)
    return SYSTEM_PROMPT_TEMPLATE.format(
        title=exp.title,
        supplies=supplies,
        steps=steps,
        knowledge_points=kps,
    )


# ------------------------------------------------------------------
#  ExperimentStore
# ------------------------------------------------------------------

class ExperimentStore:
    """实验提示词管理"""

    def __init__(self, experiments_dir: str = "experiments"):
        self._experiments: Dict[str, Experiment] = {}
        self._load_all(Path(experiments_dir))

    def _load_all(self, directory: Path) -> None:
        """扫描目录下所有 .md 文件，解析并加载"""
        if not directory.exists():
            logger.warning("实验目录不存在: %s", directory)
            return

        for md_file in sorted(directory.glob("*.md")):
            exp = _parse_experiment_md(md_file)
            if exp:
                self._experiments[exp.id] = exp
                logger.info("已加载实验: [%s] %s (%d 步骤, %d 知识点)",
                            exp.id, exp.title, len(exp.steps), len(exp.knowledge_points))

        logger.info("共加载 %d 个实验", len(self._experiments))

    def list_experiments(self) -> Dict[str, str]:
        """返回 {id: title} 映射"""
        return {eid: exp.title for eid, exp in self._experiments.items()}

    def get_experiment(self, experiment_id: str) -> Experiment:
        """按 ID 获取实验，未找到时抛出 KeyError"""
        if experiment_id not in self._experiments:
            available = ", ".join(sorted(self._experiments.keys()))
            raise KeyError(
                f"实验 '{experiment_id}' 不存在，可用: [{available}]"
            )
        return self._experiments[experiment_id]

    def build_system_prompt(self, experiment_id: str) -> str:
        """取出实验知识，格式化为完整 system prompt"""
        exp = self.get_experiment(experiment_id)
        return _format_knowledge(exp)

    def build_messages(
        self,
        experiment_id: str,
        query: str,
        rag_context: Optional[str] = None,
    ) -> List[Dict]:
        """构建 OpenAI 格式的 messages 列表（system + user）

        system prompt 包含实验知识（固定，命中前缀缓存），
        RAG 上下文作为 user message 的补充信息。
        """
        system_prompt = self.build_system_prompt(experiment_id)

        user_content = query
        if rag_context:
            user_content = f"参考资料：\n{rag_context}\n\n用户问题：{query}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = ExperimentStore("experiments")
    print("已加载实验:", store.list_experiments())
    for eid in sorted(store.list_experiments()):
        print(f"\n{'='*60}")
        prompt = store.build_system_prompt(eid)
        print(prompt[:300], "...")

