"""
名录管理与 Query 增强
=====================
启动时一次性加载 table.json 并建内存索引（中/英文名反向映射），
提供模糊查表 + BSL 等级注入 + Query 增强。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from slot_parser import LLMSlots


# ------------------------------------------------------------------
#  活动列名 → 标准 activity key 的映射
# ------------------------------------------------------------------

ACTIVITY_COLUMN_MAP = {
    # 表1
    "病毒培养":               "culture",
    "动物感染实验":           "animal",
    "未经培养的感染材料的操作": "sample",
    "灭活材料的操作":         "inactivated",
    "无感染性材料的操作":     "noninfectious",
    # 表2/3
    "活菌操作":               "culture",
    "样本检测":               "sample",
    "非感染性材料的实验":     "noninfectious",
    # 附录
    "组织培养":               "culture",
    "动物感染":               "animal",
    "感染性材料的检测":       "sample",
}


# ------------------------------------------------------------------
#  数据结构
# ------------------------------------------------------------------

@dataclass
class CatalogRecord:
    """名录中的一条记录"""
    cn_name: str
    en_name: str
    hazard_class: str
    lab_level: Dict[str, str]       # activity → BSL/ABSL 等级
    transport: Optional[Dict] = None
    notes: str = ""


# ------------------------------------------------------------------
#  CatalogManager
# ------------------------------------------------------------------

class CatalogManager:
    """名录查表 + Query 增强，启动时加载一次"""

    def __init__(self, table_json_path: str = "table.json"):
        self._catalog: Dict[str, CatalogRecord] = {}
        self._cn_index: Dict[str, str] = {}    # cn_name → key
        self._en_index: Dict[str, str] = {}    # en_name → key
        self._load(table_json_path)

    def _load(self, path: str) -> None:
        """加载 table.json 并建立中英文名反向索引"""
        # TODO: 解析 table.json，填充 self._catalog / _cn_index / _en_index
        pass

    def fuzzy_lookup(self, name: str) -> Optional[CatalogRecord]:
        """三级 fallback 模糊匹配：精确 → 包含 → 相似度

        1. 精确匹配：中文名或英文名完全一致
        2. 包含匹配：处理 LLM 输出少了/多了修饰词的情况，取最长匹配
        3. 字符串相似度：SequenceMatcher.ratio() >= 0.6

        Parameters
        ----------
        name : str
            待查找的病原体名称（中文或英文）

        Returns
        -------
        Optional[CatalogRecord]
            匹配到的记录，未匹配返回 None
        """
        # TODO: 实现三级 fallback 逻辑
        pass

    def resolve_lab_level(
        self, record: CatalogRecord, activity: Optional[str]
    ) -> Optional[str]:
        """从 CatalogRecord 提取 BSL 等级

        Parameters
        ----------
        record : CatalogRecord
            名录记录
        activity : Optional[str]
            标准化的活动类型 key（如 "culture", "sample"）

        Returns
        -------
        Optional[str]
            BSL 等级字符串，如 "BSL-2"；无法确定时返回 None
        """
        # TODO: 根据 activity 从 record.lab_level 提取等级
        pass

    def enhance_query(self, slots: LLMSlots) -> str:
        """根据槽位做名录查表 + BSL 注入，返回增强后的检索 query

        流程：
        1. 用 slots.pathogen 查名录
        2. 解析 BSL 等级
        3. 将 BSL 等级信息注入到 slots.query 中

        Parameters
        ----------
        slots : LLMSlots
            解析后的槽位

        Returns
        -------
        str
            增强后的检索 query
        """
        # TODO: 实现查表 + BSL 注入逻辑
        pass


if __name__ == "__main__":
    # 独立验证：加载 table.json，测试模糊匹配
    cm = CatalogManager("table.json")
    result = cm.fuzzy_lookup("示例病原体A")
    print(result)
