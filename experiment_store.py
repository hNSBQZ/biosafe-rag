"""
实验提示词管理
==============
启动时加载 experiments/ 目录下所有 YAML 文件到内存，
按实验 ID 查找并构建 system prompt / messages。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ------------------------------------------------------------------
#  Prompt 模板（system prompt 固定部分，命中前缀缓存）
# ------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
你是一个生物安全实验指导助手。你需要根据以下实验知识，回答用户关于实验操作、安全规范的问题。

## 实验知识
{knowledge}

## 回答要求
- 回答必须基于上述实验知识，不得编造
- 如涉及安全等级、防护要求，必须准确引用
- 如无法确定，请明确告知用户
"""


# ------------------------------------------------------------------
#  数据结构
# ------------------------------------------------------------------

@dataclass
class Experiment:
    """单个实验的元数据与知识"""
    id: str
    name: str
    knowledge: str              # ~2000 字实验场景知识
    description: str = ""


# ------------------------------------------------------------------
#  ExperimentStore
# ------------------------------------------------------------------

class ExperimentStore:
    """实验提示词管理"""

    def __init__(
        self,
        experiments_dir: str = "experiments",
        prompt_template: str = SYSTEM_PROMPT_TEMPLATE,
    ):
        self._experiments: Dict[str, Experiment] = {}
        self._prompt_template = prompt_template
        self._load_all(Path(experiments_dir))

    def _load_all(self, directory: Path) -> None:
        """扫描目录下所有 .yaml 文件，加载为 Experiment 对象"""
        # TODO: 遍历 directory，用 yaml.safe_load 解析，填充 self._experiments
        pass

    def get_experiment(self, experiment_id: str) -> Experiment:
        """按 ID 获取实验，未找到时抛出 KeyError

        Parameters
        ----------
        experiment_id : str
            实验唯一标识

        Returns
        -------
        Experiment
        """
        # TODO: 从 self._experiments 查找并返回
        pass

    def build_system_prompt(self, experiment_id: str) -> str:
        """取出实验知识，填充到 prompt 模版，返回完整 system prompt

        Parameters
        ----------
        experiment_id : str
            实验唯一标识

        Returns
        -------
        str
            填充好的完整 system prompt
        """
        # TODO: 获取实验 → 填充模板 → 返回
        pass

    def build_messages(
        self,
        experiment_id: str,
        query: str,
        rag_context: Optional[str] = None,
    ) -> List[Dict]:
        """构建完整 messages 列表（system + user）

        system prompt 固定不变（命中前缀缓存），
        RAG 上下文（每次不同）作为 user message 追加。

        Parameters
        ----------
        experiment_id : str
            实验唯一标识
        query : str
            用户原始 query
        rag_context : Optional[str]
            RAG 检索结果拼接的上下文文本

        Returns
        -------
        List[Dict]
            OpenAI 格式的 messages 列表
        """
        # TODO: 构建 [system, user(+rag_context)] 消息列表
        pass


if __name__ == "__main__":
    # 独立验证：加载 YAML，测试 build_system_prompt
    store = ExperimentStore("experiments")
    prompt = store.build_system_prompt("pcr_amplification")
    print(prompt[:200] if prompt else "No experiment loaded")
