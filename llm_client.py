"""
LLM 客户端封装
==============
对 OpenAI 兼容 API 的统一封装（适用于 DeepSeek / Qwen / 本地部署等）。
只做「调 API」这一件事，不含业务逻辑。
"""

import logging
import time
from typing import List, Dict, Optional

from openai import OpenAI

from config import LLMProfile

logger = logging.getLogger(__name__)


class LLMClient:
    """单个 LLM Profile 的客户端"""

    def __init__(self, profile: LLMProfile):
        self.profile = profile
        self._client = OpenAI(
            api_key=profile.api_key,
            base_url=profile.base_url,
            timeout=profile.timeout,
            max_retries=profile.max_retries,
        )

    # ----------------------------------------------------------
    #  文本生成
    # ----------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """单次对话，返回 assistant 回复文本"""
        resp = self._client.chat.completions.create(
            model=self.profile.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def batch_chat(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_tokens: int = 64,
        temperature: float = 0.0,
        delay: float = 0.1,
    ) -> List[str]:
        """顺序批量调用（简单实现，后续可换并发/batch API）"""
        results: List[str] = []
        for i, messages in enumerate(message_batches):
            try:
                result = self.chat(messages, max_tokens=max_tokens,
                                   temperature=temperature)
                results.append(result)
            except Exception as e:
                logger.warning("batch_chat item %d failed: %s", i, e)
                results.append("")
            if delay > 0 and i < len(message_batches) - 1:
                time.sleep(delay)
        return results

    # ----------------------------------------------------------
    #  Embedding
    # ----------------------------------------------------------

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取 embedding 向量"""
        # TODO: 实现 embedding 调用，接口参考：
        #   resp = self._client.embeddings.create(
        #       model=self.profile.model,
        #       input=texts,
        #   )
        #   return [item.embedding for item in resp.data]
        pass

    # ----------------------------------------------------------
    #  工具方法
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        return (f"LLMClient(model={self.profile.model!r}, "
                f"base_url={self.profile.base_url!r})")
