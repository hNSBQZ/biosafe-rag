"""
Chunk 后处理器
==============
负责对 split_chunk 产出的 Block/Chunk 进行：
  1. Role 打标（关键词规则 + LLM 批量兜底）
  2. 构造 Milvus 入库记录

设计参考：XR语音助手意图路由与RAG架构方案.md §七
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

from config import AppConfig, load_config
from llm_client import LLMClient, BatchTagClient, BatchEmbeddingClient
from split_chunk import Block, Chunk

logger = logging.getLogger(__name__)


# ==================================================================
#  Role 关键词规则
# ==================================================================

# 每个 role 的关键词及其权重
# ──────────────────────────────────────────────────────────────
# 设计原则：
#   关键词应反映文档的 **角色/格式**，而非领域主题。
#   例如 SOP 应靠"步骤、流程、操作规程"等程序性标志词识别，
#   而不是靠"PCR、离心、电泳"等特定实验术语——后者只说明主题，
#   一篇讲 PCR 原理的 knowledge 文章同样会包含这些词。
#   领域特定的实验/设备名词不应出现在此处，否则：
#     1. 会造成跨 role 误判（假阳性）
#     2. 对未覆盖领域的 SOP 失效（假阴性）
#
#   另：结构性特征（编号列表、步骤格式等）通过 _score_structure()
#   额外加分，不依赖关键词。
# ──────────────────────────────────────────────────────────────
ROLE_KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "sop": [
        # -- 文档类型直接标识 --
        ("标准操作规程", 4), ("操作规程", 3), ("标准操作", 3),
        ("SOP", 3), ("sop", 3),
        ("操作步骤", 3), ("操作流程", 3), ("操作方法", 2),
        ("实验步骤", 2), ("实验流程", 2), ("实验方法", 2),
        # -- 通用程序性结构词（体现"按步骤执行"） --
        ("按照以下步骤", 3), ("依次", 1), ("先后顺序", 2),
        ("然后", 1), ("接着", 1), ("随后", 1), ("完成后", 2),
        ("第一步", 2), ("第二步", 2), ("第三步", 2),
        ("准备工作", 2), ("操作要点", 2), ("操作要求", 2),
        ("注意事项", 1),
        # -- SOP 文档通用章节名 --
        ("目的与范围", 2), ("适用范围", 1), ("职责", 1),
        ("记录表", 2), ("记录表格", 2),
        # -- 通用操作动词（跨领域适用） --
        ("记录", 1), ("检查", 1), ("确认", 1),
        ("清洁", 1), ("消毒", 1), ("处理", 1),
    ],
    "emergency": [
        # -- 核心标识 --
        ("应急预案", 5), ("应急", 4), ("预案", 3),
        ("应急处置", 4), ("应急响应", 3),
        ("生物安全事故", 4),
        # -- 事故/危害场景 --
        ("泄漏", 3), ("溢洒", 3), ("暴露", 3),
        ("事故", 3), ("紧急", 3), ("处置", 2),
        ("职业暴露", 3), ("意外暴露", 3),
        # -- 应对措施 --
        ("感染", 2), ("伤害", 2), ("急救", 3),
        ("报告", 1), ("警报", 2), ("疏散", 3),
        ("消毒处理", 2),
    ],
    "regulation": [
        # -- 法规/制度类型标识 --
        ("法规", 3), ("条例", 3), ("管理办法", 3),
        ("管理制度", 3), ("管理规定", 3),
        ("制度", 2), ("规定", 2), ("合规", 2),
        ("生物安全法", 4), ("合规要求", 3),
        # -- 强制性/规范性语气词 --
        ("禁止", 2), ("不得", 2), ("应当", 1), ("必须", 1), ("要求", 1),
        # -- 行政流程 --
        ("审批", 2), ("许可", 2), ("资质", 2),
        ("备案", 2), ("报批", 2), ("监督管理", 2),
        ("安全管理", 2), ("实验室管理", 2),
        ("病原微生物实验室", 2),
    ],
    "directory": [
        # -- 核心标识 --
        ("名录", 5), ("清单", 3), ("目录", 3),
        # -- 分类分级体系 --
        ("分类", 2), ("分级", 2),
        ("第一类", 3), ("第二类", 3), ("第三类", 3), ("第四类", 3),
        ("高致病性", 3), ("人间传染", 2),
        ("BSL-1", 2), ("BSL-2", 2), ("BSL-3", 2), ("BSL-4", 2),
        # -- 微生物相关（目录/名录的主要对象） --
        ("病原微生物", 2), ("病原体", 2),
        ("菌种", 2), ("毒种", 2), ("毒株", 2),
    ],
    "knowledge": [
        # -- 认知/解释性标识 --
        ("原理", 3), ("概念", 3), ("定义", 3),
        ("术语", 3), ("术语和定义", 4),
        ("知识", 2), ("机制", 2), ("机理", 2),
        ("理论", 2), ("解释", 1),
        # -- 说明/论述性语气词 --
        ("是什么", 2), ("为什么", 2), ("因为", 1), ("由于", 1),
        ("是指", 2), ("即", 1), ("称为", 2),
        # -- 描述性属性词（跨领域通用） --
        ("结构", 1), ("功能", 1), ("特征", 1), ("特性", 1),
        ("分类", 1), ("组成", 1), ("作用", 1),
    ],
    "equipment": [
        # -- 核心标识 --
        ("设备", 3), ("仪器", 3),
        # -- 设备管理动作（体现"设备角色"的通用词） --
        ("校准", 3), ("维护", 2), ("保养", 2),
        ("故障", 3), ("维修", 2), ("巡检", 2),
        ("操作面板", 2), ("使用说明", 2),
        ("设备管理", 3), ("仪器管理", 3),
        ("开机", 2), ("关机", 2), ("运行参数", 2),
    ],
    "reagent": [
        # -- 核心标识 --
        ("试剂", 3), ("药品", 2), ("化学品", 2),
        ("试剂盒", 2), ("试剂准备", 2),
        # -- 安全信息 --
        ("MSDS", 4), ("SDS", 4), ("安全数据表", 3),
        ("安全技术说明书", 3),
        ("危害", 2), ("有毒", 2), ("腐蚀", 2), ("易燃", 2),
        # -- 存储与使用 --
        ("保存", 1), ("存储", 1), ("储存", 1),
        ("配方", 2), ("浓度", 1), ("稀释", 1),
        ("有效期", 2), ("保质期", 2),
    ],
    "notice": [
        # -- 核心标识 --
        ("通知", 4), ("公告", 4),
        # -- 行政/培训活动 --
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

# 源文件名先验已移除。
# 原因：role 打标的粒度是 chunk 级，而一篇文档通常包含多种角色的内容
# （如 通用安全.md 里同时有 knowledge/equipment/regulation 章节）。
# 文件名级的统一加分会让同一文档内不同角色的 chunk 被错误偏置。
# 章节级的信号已由 HEADING_KEYWORDS 覆盖，粒度更合适。


# ==================================================================
#  结构特征加分（不依赖领域关键词，靠文本格式判断 role）
#  参考：文档分块与RAG检索策略设计说明.md §3 structure boosts
# ==================================================================

# 编号步骤模式：匹配 "1." "（1）" "(1)" "一、" "Step 1" "第一步" 等
_STEP_PATTERN = re.compile(
    r'(?:^|\n)\s*'
    r'(?:'
    r'\d+[.、)）]'              # 1. 2、 3) 4）
    r'|[（(]\d+[)）]'           # （1） (2)
    r'|[一二三四五六七八九十]+[、.]'  # 一、 二.
    r'|第[一二三四五六七八九十\d]+步' # 第一步 第2步
    r'|[Ss]tep\s*\d+'          # Step 1 step2
    r')'
)

_DEFINITION_PATTERN = re.compile(
    r'(?:^|\n)\s*'
    r'(?:'
    r'\d+\.\d+\s+\S'           # "2.1 气溶胶" 式术语条目
    r'|术语|定义|含义|是指|即：|即,'
    r')'
)

_REGULATION_TONE_WORDS = ("必须", "应当", "不得", "禁止", "严禁", "违反")


def _score_structure(text: str) -> Dict[str, int]:
    """根据文本的排版结构特征为各 role 加分，与领域无关"""
    bonuses: Dict[str, int] = {}

    step_matches = _STEP_PATTERN.findall(text)
    if len(step_matches) >= 3:
        bonuses["sop"] = bonuses.get("sop", 0) + 4
    elif len(step_matches) >= 1:
        bonuses["sop"] = bonuses.get("sop", 0) + 2

    if _DEFINITION_PATTERN.search(text):
        bonuses["knowledge"] = bonuses.get("knowledge", 0) + 3

    tone_count = sum(text.count(w) for w in _REGULATION_TONE_WORDS)
    if tone_count >= 3:
        bonuses["regulation"] = bonuses.get("regulation", 0) + 3
    elif tone_count >= 1:
        bonuses["regulation"] = bonuses.get("regulation", 0) + 1

    return bonuses


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
    tag_client : LLMClient / BatchTagClient or None
        用于低置信度 chunk 的 LLM 打标。为 None 时仅用规则打标。
        推荐使用 BatchTagClient 以降低成本。
    emb_client : LLMClient or None
        用于 embedding 生成（当前预留）。
    config : AppConfig
        全局配置。
    """

    def __init__(
        self,
        config: AppConfig,
        tag_client: Optional[Union[LLMClient, BatchTagClient]] = None,
        emb_client: Optional[Union[LLMClient, BatchEmbeddingClient]] = None,
        use_llm: bool = True,
    ):
        self.config = config
        self.tag_client = tag_client
        self.emb_client = emb_client
        self.use_llm = use_llm

    # ----------------------------------------------------------
    #  辅助：去除 content 开头的面包屑 / 标题行
    # ----------------------------------------------------------

    @staticmethod
    def _strip_heading_prefix(content: str) -> str:
        """去除 content 开头的面包屑路径行和 markdown 标题行，只保留正文。

        split_chunk 产出的 content 结构：
          第 1 行  = 面包屑路径（如 "二．操作步骤 > 1.配置反应体系"）
          后续若干 = 空行 / markdown 标题行（# ...）
          之后     = 真正的正文
        打分 / LLM 打标时应仅使用正文，避免路径中的关键词干扰。
        """
        lines = content.split('\n')
        if len(lines) <= 1:
            return content
        start = 1                       # 跳过第 1 行（面包屑）
        for i in range(1, len(lines)):
            stripped = lines[i].strip()
            if not stripped or re.match(r'^#{1,6}\s', stripped):
                start = i + 1
                continue
            break
        body = '\n'.join(lines[start:]).strip()
        return body if body else content  # 全被剥离时回退到原文

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
                body_text = self._strip_heading_prefix(chunk.content)
                role, confidence = self.score_role(
                    body_text, heading_path, block.source_file
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

        if low_confidence and self.use_llm and self.tag_client:
            self._tag_low_confidence(low_confidence, tagged)
        elif low_confidence and not self.use_llm:
            logger.info("use_llm=False，跳过 %d 条低置信度 chunk 的 LLM 打标",
                        len(low_confidence))

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

        # 结构特征加分（编号步骤→sop, 术语条目→knowledge 等）
        for role, bonus in _score_structure(text).items():
            if role in scores:
                scores[role] += bonus

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
            body_text = self._strip_heading_prefix(tc.content)
            content_preview = body_text[:800]
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

    def embed_chunks(
        self, tagged_chunks: List[TaggedChunk],
    ) -> List[List[float]]:
        """批量生成 embedding 向量

        支持 BatchEmbeddingClient（Batch API）和 LLMClient（同步）两种后端。
        返回与 tagged_chunks 顺序一致的向量列表。
        """
        if not self.emb_client:
            raise RuntimeError("未配置 emb_client，无法生成 embedding")

        texts = [tc.content for tc in tagged_chunks]

        if isinstance(self.emb_client, BatchEmbeddingClient):
            logger.info("通过 Batch API 批量生成 embedding (%d 条)", len(texts))
            return self.emb_client.batch_embed(texts)

        logger.info("通过同步接口生成 embedding (%d 条)", len(texts))
        return self.emb_client.get_embeddings(texts)

    def to_milvus_records(
        self,
        tagged_chunks: List[TaggedChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[Dict]:
        """将 TaggedChunk 列表转换为 Milvus 可插入的 dict 列表

        Parameters
        ----------
        tagged_chunks : list of TaggedChunk
        embeddings : list of vectors, optional
            与 tagged_chunks 等长的 embedding 向量列表。
            若提供则写入 "embedding" 字段；若为 None 则不包含该字段。
        """
        if embeddings is not None and len(embeddings) != len(tagged_chunks):
            raise ValueError(
                f"embeddings 长度 ({len(embeddings)}) "
                f"与 tagged_chunks 长度 ({len(tagged_chunks)}) 不一致"
            )

        records = []
        for i, tc in enumerate(tagged_chunks):
            record = {
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
            }
            if embeddings is not None:
                record["embedding"] = embeddings[i]
            records.append(record)
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

    for role, bonus in _score_structure(text).items():
        if role in scores:
            scores[role] += bonus

    return scores


# ==================================================================
#  测试入口
# ==================================================================

def _load_blocks_from_json(path: str) -> List[Block]:
    """从 chunks_output.json 反序列化为 Block/Chunk 对象列表"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks: List[Block] = []
    for item in data:
        chunks = [
            Chunk(
                title=c["title"],
                content=c["content"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                heading_level=c["heading_level"],
                token_estimate=c.get("token_estimate", 0),
            )
            for c in item.get("chunks", [])
        ]
        blocks.append(Block(
            title=item["title"],
            chunks=chunks,
            start_line=item.get("start_line", 0),
            end_line=item.get("end_line", 0),
            heading_level=item.get("heading_level", 0),
            source_file=item.get("source_file", ""),
            breadcrumb=item.get("breadcrumb", []),
        ))
    return blocks


def main():
    import time as _time

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = _time.time()

    # ── 1. 加载配置 ──
    config = load_config()
    logger.debug("配置加载完成: threshold=%d, roles=%s",
                 config.role_confidence_threshold, config.roles)
    logger.debug("Batch 配置: poll_interval=%d, max_wait=%d, endpoint=%s",
                 config.batch.poll_interval, config.batch.max_wait,
                 config.batch.endpoint)

    # ── 2. 初始化客户端 ──
    use_llm = False
    tag_client = None
    if use_llm and "tagging" in config.llm_profiles:
        profile = config.llm_profiles["tagging"]
        tag_client = BatchTagClient(profile, config.batch)
        logger.info("打标 LLM (Batch API): model=%s, base_url=%s",
                     profile.model, profile.base_url)
    else:
        logger.warning("仅使用关键词规则打标 (use_llm=%s, profile配置=%s)",
                        use_llm, "tagging" in config.llm_profiles)

    processor = ChunkProcessor(config=config, tag_client=tag_client, use_llm=use_llm)

    # ── 3. 加载数据 ──
    input_path = "chunks_output.json"
    output_path = "role_result.json"

    logger.info("加载分块数据: %s", input_path)
    blocks = _load_blocks_from_json(input_path)
    total_chunks = sum(len(b.chunks) for b in blocks)
    logger.info("共 %d Blocks, %d Chunks", len(blocks), total_chunks)

    for b_idx, block in enumerate(blocks):
        logger.debug("  Block[%d] source=%s title='%s' breadcrumb=%s chunks=%d",
                      b_idx, block.source_file, block.title,
                      block.breadcrumb, len(block.chunks))

    # ── 4. 关键词规则打分（逐条详细输出） ──
    threshold = config.role_confidence_threshold
    tagged: List[TaggedChunk] = []
    low_confidence: List[Tuple[int, TaggedChunk]] = []

    chunk_global_idx = 0
    t_rule_start = _time.time()

    for b_idx, block in enumerate(blocks):
        block_id = f"{block.source_file}_block{b_idx}"
        heading_path = " > ".join(block.breadcrumb)

        for c_idx, chunk in enumerate(block.chunks):
            chunk_id = f"{block_id}_chunk{c_idx}"
            body_text = processor._strip_heading_prefix(chunk.content)
            role, confidence = processor.score_role(
                body_text, heading_path, block.source_file
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

            is_low = confidence < threshold
            status = "→ LLM" if is_low else "✓ 确定"
            logger.debug(
                "  [%03d] %s | role=%-12s conf=%2d %s | '%s'",
                chunk_global_idx, chunk_id, role, confidence, status,
                body_text[:60].replace('\n', ' '),
            )

            if is_low:
                low_confidence.append((chunk_global_idx, tc))
            tagged.append(tc)
            chunk_global_idx += 1

    t_rule_end = _time.time()
    rule_decided = len(tagged) - len(low_confidence)
    logger.info("规则打分完成: %.2fs | 规则确定 %d, 低置信度待 LLM %d",
                t_rule_end - t_rule_start, rule_decided, len(low_confidence))

    # ── 5. LLM 批量打标 ──
    if low_confidence and use_llm and tag_client:
        logger.info("开始 LLM Batch 打标 (%d 条)...", len(low_confidence))
        t_llm_start = _time.time()
        processor._tag_low_confidence(low_confidence, tagged)
        t_llm_end = _time.time()
        logger.info("LLM 打标完成: %.2fs", t_llm_end - t_llm_start)
    elif low_confidence and not use_llm:
        logger.info("use_llm=False，跳过 %d 条低置信度 chunk 的 LLM 打标",
                     len(low_confidence))

        for _, tc in low_confidence:
            logger.debug(
                "  LLM 结果: %s | role=%-12s tagged_by=%s | '%s'",
                tc.chunk_id, tc.role, tc.tagged_by,
                tc.content[:60].replace('\n', ' '),
            )
    elif low_confidence:
        logger.warning("有 %d 条低置信度 chunk 但无 tag_client，跳过 LLM 打标",
                        len(low_confidence))

    # ── 6. 汇总统计 ──
    rule_count = sum(1 for t in tagged if t.tagged_by == "rule")
    llm_count = sum(1 for t in tagged if t.tagged_by == "llm")
    logger.info("打标汇总: 总计 %d | 规则 %d | LLM %d",
                len(tagged), rule_count, llm_count)

    # ── 7. 导出 ──
    records = processor.to_milvus_records(tagged)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    t_end = _time.time()
    logger.info("结果已写入: %s (%d 条)", output_path, len(records))
    logger.info("总耗时: %.2fs", t_end - t_start)

    # ── 8. 打印摘要 ──
    role_dist: Dict[str, int] = {}
    for tc in tagged:
        role_dist[tc.role] = role_dist.get(tc.role, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"打标完成: {len(tagged)} chunks (规则: {rule_count}, LLM: {llm_count})")
    print(f"总耗时: {t_end - t_start:.2f}s")
    print(f"Role 分布:")
    for role, count in sorted(role_dist.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {role:12s} {count:3d} {bar}")
    print(f"结果: {output_path}")


if __name__ == "__main__":
    main()
