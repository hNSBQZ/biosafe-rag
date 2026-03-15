"""
RAG 合成器
==========
将检索到的 chunk 拼接为上下文，调用 LLM 第二轮生成最终回答。

职责单一：上下文组装 + LLM 调用，不含检索逻辑。
"""

import logging
from typing import Dict, List, Optional

from experiment_store import ExperimentStore
from llm_client import LLMClient

logger = logging.getLogger(__name__)

_RAG_USER_TEMPLATE = """\
以下是与用户问题相关的参考资料，请结合这些资料和你已有的实验知识回答。
如果参考资料不足以回答，请如实说明，不要编造。

---参考资料开始---
{rag_context}
---参考资料结束---

用户问题：{query}"""


class RAGSynthesizer:
    """RAG 第二轮合成：上下文拼接 + LLM 生成最终回答"""

    def __init__(
        self,
        llm_client: LLMClient,
        experiment_store: ExperimentStore,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        enable_thinking: bool = False,
    ):
        self._llm = llm_client
        self._experiments = experiment_store
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._enable_thinking = enable_thinking

    def synthesize(
        self,
        query: str,
        chunks: List[Dict],
        experiment_id: str,
        first_round_answer: Optional[str] = None,
    ) -> Dict:
        """将 chunks 拼为上下文，调 LLM 第二轮生成最终回答

        Parameters
        ----------
        query : str
            原始用户 query
        chunks : List[Dict]
            Retriever 返回的 chunk 列表（已经过升格 + 预算裁剪）
        experiment_id : str
            当前实验 ID（用于加载 system prompt，命中前缀缓存）
        first_round_answer : str, optional
            第一轮 LLM 的回答（如有），可作为兜底

        Returns
        -------
        Dict
            type="rag", content=最终回答, sources=chunk_id 列表
        """
        rag_context = self._assemble_context(chunks)
        logger.info(
            "RAG 合成 | chunks=%d | context_len=%d chars",
            len(chunks), len(rag_context),
        )

        user_content = _RAG_USER_TEMPLATE.format(
            rag_context=rag_context,
            query=query,
        )

        system_prompt = self._experiments.build_system_prompt(experiment_id)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        extra_body = None
        if not self._enable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        logger.info("LLM 第二轮调用 | experiment=%s", experiment_id)
        answer = self._llm.chat(
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            extra_body=extra_body,
        )
        logger.info("LLM 第二轮完成 | answer_len=%d", len(answer))

        if not answer.strip() and first_round_answer:
            logger.warning("第二轮输出为空，回退第一轮回答")
            answer = first_round_answer

        source_ids = []
        for c in chunks:
            if c.get("promoted") and c.get("chunk_ids"):
                source_ids.extend(c["chunk_ids"])
            else:
                source_ids.append(c.get("chunk_id", ""))

        return {
            "type": "rag",
            "content": answer,
            "sources": source_ids,
            "experiment_id": experiment_id,
            "chunk_count": len(chunks),
        }

    @staticmethod
    def _assemble_context(chunks: List[Dict]) -> str:
        """将 chunk 列表拼接为带编号的上下文字符串"""
        parts = []
        for i, c in enumerate(chunks):
            heading = c.get("heading_path", "")
            content = c.get("content", "")
            source = c.get("source_file", "")
            header = f"[{i+1}] {heading}" if heading else f"[{i+1}]"
            if source:
                header += f" ({source})"
            parts.append(f"{header}\n{content}")
        return "\n\n---\n\n".join(parts)
