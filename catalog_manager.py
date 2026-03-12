"""
名录管理与 Query 增强
=====================
启动时一次性加载 table.json 并建内存索引（中/英文名反向映射），
提供模糊查表 + BSL 等级注入 + Query 增强。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from slot_parser import LLMSlots

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#  活动列名 → 标准 activity key 的映射
# ------------------------------------------------------------------

ACTIVITY_COLUMN_MAP = {
    # 表1 病毒
    "病毒培养":               "culture",
    "动物感染实验":           "animal",
    "未经培养的感染材料的操作": "sample",
    "灭活材料的操作":         "inactivated",
    "无感染性材料的操作":     "noninfectious",
    # 表2/3 细菌、真菌
    "活菌操作":               "culture",
    "样本检测":               "sample",
    "非感染性材料的实验":     "noninfectious",
    # 附录 朊病毒
    "组织培养":               "culture",
    "动物感染":               "animal",
    "感染性材料的检测":       "sample",
}

# 四张表的名称列字段名模式
_CN_NAME_KEYS = ["病毒名称/中文名", "疾病中文名", "病原菌名称/中文名", "真菌名称/中文名"]
_EN_NAME_KEYS = ["病毒名称/英文名", "疾病英文名", "病原菌名称/拉丁文名称", "真菌名称/拉丁文名称"]

# 清理名录中文名的尾注标记（如 "新型冠状病毒i" → "新型冠状病毒"）
_FOOTNOTE_RE = re.compile(r'[a-z]+$')


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
        self._cn_index: Dict[str, str] = {}    # cn_name_lower → catalog key
        self._en_index: Dict[str, str] = {}    # en_name_lower → catalog key
        self._load(table_json_path)

    # ----------------------------------------------------------
    #  加载
    # ----------------------------------------------------------

    def _load(self, path: str) -> None:
        """加载 table.json 并建立中英文名反向索引"""
        with open(path, "r", encoding="utf-8") as f:
            tables = json.load(f)

        for table in tables:
            columns = table.get("columns", [])
            records = table.get("records", [])

            cn_key = self._find_column(columns, _CN_NAME_KEYS)
            en_key = self._find_column(columns, _EN_NAME_KEYS)
            if not cn_key:
                logger.warning("表 '%s' 找不到中文名列，跳过", table.get("table_name"))
                continue

            activity_map = self._build_activity_map(columns)

            for rec in records:
                cn_raw = rec.get(cn_key, "").strip()
                if not cn_raw:
                    continue

                cn_name = _FOOTNOTE_RE.sub("", cn_raw).strip()
                en_name = rec.get(en_key, "").strip() if en_key else ""
                hazard_class = rec.get("危害程度分类", "").strip()

                lab_level: Dict[str, str] = {}
                for col_name, activity_key in activity_map.items():
                    level = rec.get(col_name, "").strip()
                    if level:
                        lab_level[activity_key] = level

                transport = None
                transport_ab = None
                transport_un = None
                for col in columns:
                    if "A/B" in col:
                        transport_ab = rec.get(col, "").strip()
                    if "UN编号" in col:
                        transport_un = rec.get(col, "").strip()
                if transport_ab or transport_un:
                    transport = {"category": transport_ab or "", "un_number": transport_un or ""}

                notes = rec.get("备注", "").strip()

                catalog_key = cn_name
                record = CatalogRecord(
                    cn_name=cn_name,
                    en_name=en_name,
                    hazard_class=hazard_class,
                    lab_level=lab_level,
                    transport=transport,
                    notes=notes,
                )
                self._catalog[catalog_key] = record
                self._cn_index[cn_name] = catalog_key
                if en_name:
                    self._en_index[en_name.lower()] = catalog_key

        logger.info("名录加载完成: %d 条记录", len(self._catalog))

    @staticmethod
    def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
        """从列名列表中找到第一个匹配的候选列名"""
        for c in candidates:
            if c in columns:
                return c
        return None

    @staticmethod
    def _build_activity_map(columns: List[str]) -> Dict[str, str]:
        """从列名中提取 '实验活动所需实验室等级/XXX' 列并映射到标准 activity key"""
        prefix = "实验活动所需实验室等级/"
        result: Dict[str, str] = {}
        for col in columns:
            if not col.startswith(prefix):
                continue
            raw_activity = col[len(prefix):]
            # 去掉列名末尾的脚注字母（如 "病毒培养a" → "病毒培养"）
            clean = re.sub(r'[a-zA-Z]+$', '', raw_activity).strip()
            if clean in ACTIVITY_COLUMN_MAP:
                result[col] = ACTIVITY_COLUMN_MAP[clean]
        return result

    # ----------------------------------------------------------
    #  模糊查表
    # ----------------------------------------------------------

    def fuzzy_lookup(self, name: str) -> Optional[CatalogRecord]:
        """三级 fallback 模糊匹配：精确 → 包含 → 相似度

        Parameters
        ----------
        name : str
            待查找的病原体名称（中文或英文）

        Returns
        -------
        Optional[CatalogRecord]
            匹配到的记录，未匹配返回 None
        """
        if not name:
            return None

        name_norm = name.strip()

        # Level 1: 精确匹配
        if name_norm in self._cn_index:
            return self._catalog[self._cn_index[name_norm]]
        en_lower = name_norm.lower()
        if en_lower in self._en_index:
            return self._catalog[self._en_index[en_lower]]

        # Level 2: 包含匹配（处理 LLM 输出少了/多了修饰词的情况）
        candidates: List[tuple] = []
        for rec in self._catalog.values():
            if name_norm in rec.cn_name or rec.cn_name in name_norm:
                candidates.append((rec, len(rec.cn_name)))
            elif en_lower in rec.en_name.lower() or rec.en_name.lower() in en_lower:
                if rec.en_name:
                    candidates.append((rec, len(rec.en_name)))
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            match = candidates[0][0]
            logger.debug("包含匹配: '%s' → '%s'", name_norm, match.cn_name)
            return match

        # Level 3: 字符串相似度
        best_score, best_rec = 0.0, None
        for rec in self._catalog.values():
            score = SequenceMatcher(None, name_norm, rec.cn_name).ratio()
            if score > best_score:
                best_score, best_rec = score, rec
            if rec.en_name:
                en_score = SequenceMatcher(None, en_lower, rec.en_name.lower()).ratio()
                if en_score > best_score:
                    best_score, best_rec = en_score, rec
        if best_score >= 0.6:
            logger.debug("相似度匹配: '%s' → '%s' (score=%.2f)",
                         name_norm, best_rec.cn_name, best_score)
            return best_rec

        logger.debug("名录未匹配: '%s'", name_norm)
        return None

    # ----------------------------------------------------------
    #  BSL 等级解析
    # ----------------------------------------------------------

    def resolve_lab_level(
        self, record: CatalogRecord, activity: Optional[str]
    ) -> Optional[str]:
        """从 CatalogRecord 提取 BSL 等级

        activity 已知 → 精确返回对应等级
        activity 未知 → 返回所有活动类型的等级并集（去掉 BSL-1）
        """
        if activity and activity in record.lab_level:
            return record.lab_level[activity]

        levels = set()
        for v in record.lab_level.values():
            # 清理脚注标记如 "BSL-2f" → "BSL-2"
            clean = re.sub(r'[a-z]+$', '', v.strip())
            if clean and clean != "BSL-1":
                levels.add(clean)
        return " ".join(sorted(levels)) if levels else None

    # ----------------------------------------------------------
    #  Query 增强
    # ----------------------------------------------------------

    def enhance_query(self, slots: LLMSlots) -> str:
        """根据槽位做名录查表 + BSL 注入，返回增强后的检索 query

        流程：
        1. PATHOGEN 分支：模糊查名录 → 注入正式名 + BSL 等级
        2. EQUIPMENT 分支：暂无设备名录，跳过
        3. 拼接 slots.query

        当 roles 包含 directory 时跳过 BSL 注入——名录查询本身就是
        要查分类信息，不需要用 BSL 等级缩窄检索范围。
        """
        parts: List[str] = []
        skip_bsl = "directory" in slots.roles

        # ── 分支 A：PATHOGEN → 模糊查名录 → 注入 BSL 等级 ──
        if slots.pathogen:
            match = self.fuzzy_lookup(slots.pathogen)
            if match:
                parts.append(match.cn_name)
                if not skip_bsl:
                    lab_level = self.resolve_lab_level(match, slots.activity)
                    if lab_level:
                        parts.append(lab_level)
                else:
                    lab_level = None
                logger.info(
                    "名录命中: '%s' → %s (%s) | BSL=%s%s",
                    slots.pathogen, match.cn_name, match.hazard_class, lab_level,
                    " (directory, 跳过BSL注入)" if skip_bsl else "",
                )
            else:
                logger.info("名录未命中: '%s'", slots.pathogen)

        # ── 分支 B：EQUIPMENT → 暂无设备名录 ──
        # 预留，后续接入设备名录后在此扩展

        parts.append(slots.query or "")
        return " ".join(p for p in parts if p)

    # ----------------------------------------------------------
    #  辅助
    # ----------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._catalog)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("catalog_manager 独立验证")
    print("=" * 60)

    cm = CatalogManager("table.json")
    print(f"\n已加载 {cm.size} 条名录记录")

    # Case 1: 精确匹配
    print("\n--- Case 1: 精确匹配 ---")
    rec = cm.fuzzy_lookup("新型冠状病毒")
    if rec:
        print(f"  cn_name: {rec.cn_name}")
        print(f"  en_name: {rec.en_name}")
        print(f"  hazard_class: {rec.hazard_class}")
        print(f"  lab_level: {rec.lab_level}")
        level = cm.resolve_lab_level(rec, "noninfectious")
        print(f"  resolve(noninfectious): {level}")
        level_all = cm.resolve_lab_level(rec, None)
        print(f"  resolve(None): {level_all}")
    else:
        print("  未找到！")

    # Case 2: 包含匹配（少了修饰词）
    print("\n--- Case 2: 包含匹配 ---")
    rec2 = cm.fuzzy_lookup("炭疽杆菌")
    if rec2:
        print(f"  '炭疽杆菌' → {rec2.cn_name}")
    else:
        print("  未找到")

    # Case 3: 英文名匹配
    print("\n--- Case 3: 英文名匹配 ---")
    rec3 = cm.fuzzy_lookup("SARS-CoV-2")
    if rec3:
        print(f"  'SARS-CoV-2' → {rec3.cn_name}")
    else:
        print("  未找到")

    # Case 4: enhance_query 完整测试
    print("\n--- Case 4: enhance_query ---")
    test_slots = LLMSlots(
        needs_rag=True,
        roles=["directory", "regulation"],
        pathogen="新型冠状病毒",
        activity="noninfectious",
        query="新型冠状病毒 病原体分类 生物安全等级",
    )
    enhanced = cm.enhance_query(test_slots)
    print(f"  增强后 query: {enhanced}")

    # Case 5: 无 PATHOGEN 的 query
    print("\n--- Case 5: 无实体增强 ---")
    test_slots2 = LLMSlots(
        needs_rag=True,
        roles=["sop"],
        query="实验室废液 处理 流程 规范",
    )
    enhanced2 = cm.enhance_query(test_slots2)
    print(f"  增强后 query: {enhanced2}")

    print(f"\n{'=' * 60}")
    print("验证完成 ✓")
