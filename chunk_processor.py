"""
Chunk 后处理器
==============
负责对 split_chunk 产出的 Block/Chunk 进行：
  1. Role 打标（关键词规则 + LLM 批量兜底）
  2. 构造 Milvus 入库记录

设计参考：XR语音助手意图路由与RAG架构方案.md §七
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from config import AppConfig
from llm_client import LLMClient
from split_chunk import Block, Chunk

logger = logging.getLogger(__name__)


# ==================================================================
#  Role 关键词规则
# ==================================================================

# 每个 role 的关键词及其权重
# 权重含义：出现一次加多少分。面包屑/标题命中用 _HEADING_KEYWORDS 额外加分。
ROLE_KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "sop": [
        ("操作步骤", 3), ("操作流程", 3), ("标准操作", 3), ("操作规程", 3),
        ("SOP", 3), ("sop", 3),
        ("实验步骤", 2), ("实验流程", 2), ("操作方法", 2), ("操作要点", 2),
        ("配置反应", 2), ("反应体系", 2),
        ("加入", 1), ("移液", 1), ("离心", 2), ("电泳", 2),
        ("取样", 1), ("混匀", 1), ("震荡", 1), ("孵育", 1), ("培养", 1),
        ("扩增", 2), ("提取", 1), ("灭菌", 1), ("接种", 1),
        ("PCR", 2), ("核酸提取", 2), ("琼脂糖", 1),
        ("按照以下步骤", 2), ("依次", 1), ("先后顺序", 2),
    ],
    "emergency": [
        ("应急", 4), ("预案", 3), ("应急预案", 5),
        ("泄漏", 3), ("溢洒", 3), ("暴露", 3),
        ("事故", 3), ("紧急", 3), ("处置", 2),
        ("感染", 2), ("伤害", 2), ("急救", 3),
        ("报告", 1), ("警报", 2), ("疏散", 3),
        ("消毒处理", 2), ("生物安全事故", 4),
        ("应急处置", 4), ("应急响应", 3),
        ("职业暴露", 3), ("意外暴露", 3),
    ],
    "regulation": [
        ("法规", 3), ("制度", 2), ("管理办法", 3), ("条例", 3),
        ("规定", 2), ("合规", 2), ("要求", 1),
        ("审批", 2), ("许可", 2), ("资质", 2),
        ("禁止", 2), ("应当", 1), ("必须", 1), ("不得", 2),
        ("管理制度", 3), ("管理规定", 3),
        ("生物安全法", 4), ("病原微生物实验室", 2),
        ("安全管理", 2), ("实验室管理", 2),
        ("合规要求", 3), ("监督管理", 2),
        ("备案", 2), ("报批", 2),
    ],
    "directory": [
        ("名录", 5), ("清单", 3), ("目录", 3),
        ("分类", 2), ("分级", 2),
        ("病原微生物", 2), ("病原体", 2),
        ("第一类", 3), ("第二类", 3), ("第三类", 3), ("第四类", 3),
        ("高致病性", 3), ("人间传染", 2),
        ("BSL-1", 2), ("BSL-2", 2), ("BSL-3", 2), ("BSL-4", 2),
        ("菌种", 2), ("毒种", 2), ("毒株", 2),
    ],
    "knowledge": [
        ("原理", 3), ("概念", 3), ("定义", 3),
        ("知识", 2), ("机制", 2), ("机理", 2),
        ("理论", 2), ("解释", 1),
        ("是什么", 2), ("为什么", 2), ("因为", 1), ("由于", 1),
        ("基因", 1), ("蛋白", 1), ("细胞", 1), ("免疫", 1),
        ("结构", 1), ("功能", 1), ("特征", 1), ("特性", 1),
    ],
    "equipment": [
        ("设备", 3), ("仪器", 3),
        ("校准", 3), ("维护", 2), ("保养", 2),
        ("故障", 3), ("维修", 2),
        ("生物安全柜", 3), ("离心机", 2), ("PCR仪", 2),
        ("高压灭菌器", 2), ("超净工作台", 2), ("恒温箱", 2),
        ("操作面板", 2), ("使用说明", 2),
    ],
    "reagent": [
        ("试剂", 3), ("药品", 2), ("化学品", 2),
        ("MSDS", 4), ("安全数据表", 3),
        ("保存", 1), ("存储", 1), ("储存", 1),
        ("危害", 2), ("有毒", 2), ("腐蚀", 2), ("易燃", 2),
        ("配方", 2), ("浓度", 1), ("稀释", 1),
        ("buffer", 1), ("Buffer", 1), ("缓冲液", 1),
        ("试剂盒", 2), ("试剂准备", 2),
    ],
    "notice": [
        ("通知", 4), ("公告", 4),
        ("培训", 3), ("考核", 3),
        ("安排", 2), ("时间", 1), ("日期", 1),
        ("会议", 2), ("签到", 2), ("报名", 2),
        ("年度", 1), ("季度", 1),
    ],
}

HEADING_KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "sop": [
        ("操作", 3), ("步骤", 3), ("流程", 3), ("SOP", 4), ("sop", 4),
        ("操作规程", 4), ("标准操作", 4),
    ],
    "emergency": [
        ("应急", 5), ("预案", 4), ("事故", 3), ("处置", 3),
    ],
    "regulation": [
        ("法规", 4), ("制度", 3), ("规定", 3), ("管理", 2),
        ("要求", 2), ("安全", 1),
    ],
    "directory": [
        ("名录", 5), ("目录", 4), ("清单", 4), ("分类", 3),
    ],
    "knowledge": [
        ("原理", 4), ("知识", 3), ("概念", 3),
    ],
    "equipment": [
        ("设备", 4), ("仪器", 4),
    ],
    "reagent": [
        ("试剂", 4), ("药品", 3), ("MSDS", 5),
    ],
    "notice": [
        ("通知", 5), ("公告", 4), ("培训", 3),
    ],
}

# 源文件名到 role 的强先验（文件名本身就是很强的信号）
SOURCE_FILE_ROLE_HINT: Dict[str, Tuple[str, int]] = {
    "pcr-sop": ("sop", 6),
    "sop": ("sop", 6),
    "应急预案": ("emergency", 6),
    "应急": ("emergency", 5),
    "名录": ("directory", 6),
    "通用安全": ("regulation", 4),
    "核酸": ("sop", 3),
}


# ==================================================================
#  数据结构
# ==================================================================

@dataclass
class TaggedChunk:
    """打标后的 Chunk，准备入库"""
    chunk_id: str
    content: str
    role: str
    role_confidence: int
    block_id: str
    source_file: str
    breadcrumb: List[str]
    heading_path: str
    start_line: int
    end_line: int
    token_estimate: int
    tagged_by: str  # "rule" 或 "llm"


# ==================================================================
#  ChunkProcessor
# ==================================================================

class ChunkProcessor:
    """Chunk 后处理器：打标 + 构造入库记录

    Parameters
    ----------
    tag_client : LLMClient or None
        用于低置信度 chunk 的 LLM 打标。为 None 时仅用规则打标。
    emb_client : LLMClient or None
        用于 embedding 生成（当前预留）。
    config : AppConfig
        全局配置。
    """

    def __init__(
        self,
        config: AppConfig,
        tag_client: Optional[LLMClient] = None,
        emb_client: Optional[LLMClient] = None,
    ):
        self.config = config
        self.tag_client = tag_client
        self.emb_client = emb_client

    # ----------------------------------------------------------
    #  主入口
    # ----------------------------------------------------------

    def process_blocks(self, blocks: List[Block]) -> List[TaggedChunk]:
        """处理 Block 列表，返回打标后的 TaggedChunk 列表

        流程：
          1. 遍历所有 Block/Chunk，用关键词规则打分
          2. 置信度高的直接确定 role
          3. 置信度低的收集起来，批量调 LLM
          4. 合并结果返回
        """
        threshold = self.config.role_confidence_threshold
        tagged: List[TaggedChunk] = []
        low_confidence: List[Tuple[int, TaggedChunk]] = []

        chunk_global_idx = 0
        for b_idx, block in enumerate(blocks):
            block_id = f"{block.source_file}_block{b_idx}"
            heading_path = " > ".join(block.breadcrumb)

            for c_idx, chunk in enumerate(block.chunks):
                chunk_id = f"{block_id}_chunk{c_idx}"
                role, confidence = self.score_role(
                    chunk.content, heading_path, block.source_file
                )

                tc = TaggedChunk(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    role=role,
                    role_confidence=confidence,
                    block_id=block_id,
                    source_file=block.source_file,
                    breadcrumb=list(block.breadcrumb),
                    heading_path=heading_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    token_estimate=chunk.token_estimate,
                    tagged_by="rule",
                )

                if confidence >= threshold:
                    tagged.append(tc)
                else:
                    low_confidence.append((chunk_global_idx, tc))
                    tagged.append(tc)

                chunk_global_idx += 1

        if low_confidence and self.tag_client:
            self._tag_low_confidence(low_confidence, tagged)

        rule_count = sum(1 for t in tagged if t.tagged_by == "rule")
        llm_count = sum(1 for t in tagged if t.tagged_by == "llm")
        logger.info(
            "打标完成: %d chunks (规则: %d, LLM: %d, 低置信未处理: %d)",
            len(tagged), rule_count, llm_count,
            len(low_confidence) - llm_count,
        )

        return tagged

    # ----------------------------------------------------------
    #  关键词规则打分
    # ----------------------------------------------------------

    def score_role(
        self,
        text: str,
        heading_path: str,
        source_file: str = "",
    ) -> Tuple[str, int]:
        """对单个 chunk 的 role 进行关键词打分

        Returns
        -------
        (best_role, confidence)
            confidence = top1_score - top2_score
        """
        scores: Dict[str, int] = {r: 0 for r in self.config.roles}

        for role, keywords in ROLE_KEYWORDS.items():
            for kw, weight in keywords:
                count = text.count(kw)
                if count > 0:
                    scores[role] += weight * min(count, 3)

        for role, keywords in HEADING_KEYWORDS.items():
            for kw, weight in keywords:
                if kw in heading_path:
                    scores[role] += weight * 2

        if source_file:
            stem = source_file.rsplit(".", 1)[0] if "." in source_file else source_file
            for pattern, (role, bonus) in SOURCE_FILE_ROLE_HINT.items():
                if pattern in stem:
                    scores[role] += bonus
                    break

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top1_role, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0

        confidence = top1_score - top2_score
        return top1_role, confidence

    # ----------------------------------------------------------
    #  LLM 批量打标
    # ----------------------------------------------------------

    def _tag_low_confidence(
        self,
        low_items: List[Tuple[int, TaggedChunk]],
        all_tagged: List[TaggedChunk],
    ) -> None:
        """用 LLM 对低置信度 chunk 批量打标"""
        if not self.tag_client:
            return

        prompt_template = self.config.prompts.get("role_tagging", "")
        if not prompt_template:
            logger.warning("未找到 role_tagging prompt 模板，跳过 LLM 打标")
            return

        valid_roles = set(self.config.roles)

        message_batches = []
        for _, tc in low_items:
            content_preview = tc.content[:800]
            prompt = prompt_template.format(
                heading_path=tc.heading_path,
                chunk_text=content_preview,
            )
            message_batches.append([
                {"role": "user", "content": prompt},
            ])

        logger.info("LLM 批量打标: %d 条低置信度 chunk", len(message_batches))
        results = self.tag_client.batch_chat(
            message_batches, max_tokens=32, temperature=0.0,
        )

        for (_, tc), raw_result in zip(low_items, results):
            role = self._parse_role_result(raw_result, valid_roles)
            if role:
                tc.role = role
                tc.tagged_by = "llm"

    @staticmethod
    def _parse_role_result(raw: str, valid_roles: set) -> Optional[str]:
        """从 LLM 返回文本中提取合法 role"""
        if not raw:
            return None
        cleaned = raw.strip().lower()
        cleaned = re.sub(r'[^a-z]', '', cleaned)
        if cleaned in valid_roles:
            return cleaned
        for role in valid_roles:
            if role in raw.lower():
                return role
        return None

    # ----------------------------------------------------------
    #  转换为 Milvus 入库格式
    # ----------------------------------------------------------

    def to_milvus_records(self, tagged_chunks: List[TaggedChunk]) -> List[Dict]:
        """将 TaggedChunk 列表转换为 Milvus 可插入的 dict 列表

        注意：embedding 字段需要单独填充（调用 emb_client）。
        当前版本不含 embedding，仅构造 metadata 部分。
        """
        records = []
        for tc in tagged_chunks:
            records.append({
                "chunk_id": tc.chunk_id,
                "content": tc.content,
                "role": tc.role,
                "role_confidence": tc.role_confidence,
                "block_id": tc.block_id,
                "source_file": tc.source_file,
                "breadcrumb": tc.breadcrumb,
                "heading_path": tc.heading_path,
                "start_line": tc.start_line,
                "end_line": tc.end_line,
                "token_estimate": tc.token_estimate,
                "tagged_by": tc.tagged_by,
                # "embedding": [],  # TODO: 由 pipeline 调用 emb_client 填充
            })
        return records


# ==================================================================
#  便捷函数（供外部直接调用，如 Query 侧的关键词兜底）
# ==================================================================

def score_role(text: str, heading_path: str, source_file: str = "") -> Dict[str, int]:
    """独立的关键词打分函数，返回所有 role 的分数 dict

    供 Query 侧 keyword_role_classify 复用。
    """
    from config import AppConfig
    dummy_config = AppConfig()
    scores: Dict[str, int] = {r: 0 for r in dummy_config.roles}

    for role, keywords in ROLE_KEYWORDS.items():
        for kw, weight in keywords:
            count = text.count(kw)
            if count > 0:
                scores[role] += weight * min(count, 3)

    for role, keywords in HEADING_KEYWORDS.items():
        for kw, weight in keywords:
            if kw in heading_path:
                scores[role] += weight * 2

    if source_file:
        stem = source_file.rsplit(".", 1)[0] if "." in source_file else source_file
        for pattern, (r, bonus) in SOURCE_FILE_ROLE_HINT.items():
            if pattern in stem:
                scores[r] += bonus
                break

    return scores
