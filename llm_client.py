"""
LLM 客户端封装
==============
对 OpenAI 兼容 API 的统一封装（适用于 DeepSeek / Qwen / 本地部署等）。
只做「调 API」这一件事，不含业务逻辑。

包含三种客户端：
  - LLMClient: 同步逐条调用，适用于少量请求或在线场景
  - BatchTagClient: 基于 Batch API 的异步批量 Chat 调用（打标）
  - BatchEmbeddingClient: 基于 Batch API 的异步批量 Embedding 调用
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI

from config import LLMProfile, BatchConfig

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
        extra_body: Optional[Dict] = None,
    ) -> str:
        """单次对话，返回 assistant 回复文本"""
        kwargs = dict(
            model=self.profile.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return content.strip() if content else ""

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


class BatchTagClient:
    """基于 Dashscope Batch API 的批量打标客户端

    与 LLMClient.batch_chat 接口兼容，可直接替换。
    通过上传 JSONL → 创建 Batch 任务 → 轮询等待 → 下载结果
    实现大批量请求的异步处理，成本约为同步调用的 50%。

    Parameters
    ----------
    profile : LLMProfile
        LLM 配置（api_key / base_url / model）。
    batch_config : BatchConfig
        Batch API 运行参数（轮询间隔、超时、endpoint）。
    """

    def __init__(
        self,
        profile: LLMProfile,
        batch_config: BatchConfig = BatchConfig(),
    ):
        self.profile = profile
        self.batch_config = batch_config
        self._client = OpenAI(
            api_key=profile.api_key,
            base_url=profile.base_url,
            timeout=profile.timeout,
            max_retries=profile.max_retries,
        )

    # ----------------------------------------------------------
    #  公开接口：与 LLMClient.batch_chat 签名一致
    # ----------------------------------------------------------

    def batch_chat(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_tokens: int = 64,
        temperature: float = 0.0,
        delay: float = 0.0,
    ) -> List[str]:
        """通过 Batch API 批量调用，返回与输入顺序一致的回复列表

        delay 参数保留以兼容 LLMClient 接口，实际不使用。
        """
        if not message_batches:
            return []

        remote_file_ids: List[str] = []
        jsonl_path = self._build_jsonl(message_batches, max_tokens, temperature)
        try:
            input_file_id = self._upload(jsonl_path)
            remote_file_ids.append(input_file_id)
            batch_id = self._create_batch(input_file_id)
            self._wait_for_completion(batch_id)
            results, extra_file_ids = self._download_results(
                batch_id, len(message_batches),
            )
            remote_file_ids.extend(extra_file_ids)
            return results
        finally:
            Path(jsonl_path).unlink(missing_ok=True)
            self._cleanup_remote_files(remote_file_ids)

    # ----------------------------------------------------------
    #  内部方法
    # ----------------------------------------------------------

    def _build_jsonl(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """将 message_batches 写入临时 JSONL 文件"""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
        )
        for idx, messages in enumerate(message_batches):
            record = {
                "custom_id": str(idx),
                "method": "POST",
                "url": self.batch_config.endpoint,
                "body": {
                    "model": self.profile.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            }
            tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.close()
        logger.debug("Batch JSONL 文件已生成: %s (%d 条)", tmp.name, len(message_batches))
        return tmp.name

    def _upload(self, jsonl_path: str) -> str:
        file_obj = self._client.files.create(
            file=Path(jsonl_path), purpose="batch",
        )
        logger.info("Batch 文件已上传, file_id=%s", file_obj.id)
        return file_obj.id

    def _create_batch(self, input_file_id: str) -> str:
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint=self.batch_config.endpoint,
            completion_window="24h",
        )
        logger.info("Batch 任务已创建, batch_id=%s", batch.id)
        return batch.id

    def _wait_for_completion(self, batch_id: str) -> None:
        elapsed = 0
        poll_count = 0
        while elapsed < self.batch_config.max_wait:
            batch = self._client.batches.retrieve(batch_id=batch_id)
            status = batch.status
            poll_count += 1

            completed = getattr(batch.request_counts, "completed", "?")
            failed = getattr(batch.request_counts, "failed", "?")
            total = getattr(batch.request_counts, "total", "?")
            logger.info(
                "[轮询 #%d] Batch %s | 状态: %s | 进度: %s/%s 完成, %s 失败 | 已等待 %ds",
                poll_count, batch_id, status, completed, total, failed, elapsed,
            )

            if status == "completed":
                logger.info("Batch %s 已完成 (共轮询 %d 次, 耗时 %ds)",
                            batch_id, poll_count, elapsed)
                return
            if status in ("failed", "expired", "cancelled"):
                raise RuntimeError(
                    f"Batch 任务 {status}: {batch.errors}"
                )

            time.sleep(self.batch_config.poll_interval)
            elapsed += self.batch_config.poll_interval

        raise TimeoutError(
            f"Batch 任务 {batch_id} 超时 ({self.batch_config.max_wait}s)"
        )

    def _download_results(
        self, batch_id: str, expected_count: int,
    ) -> tuple:
        """下载并解析结果，按 custom_id 排序返回

        Returns (ordered_results, file_ids_to_cleanup)
        """
        batch = self._client.batches.retrieve(batch_id=batch_id)
        file_ids: List[str] = []

        results_by_id: Dict[str, str] = {}

        if batch.output_file_id:
            file_ids.append(batch.output_file_id)
            logger.info("下载输出文件: %s", batch.output_file_id)
            content = self._client.files.content(batch.output_file_id)
            for line in content.text.strip().split("\n"):
                if not line.strip():
                    continue
                record = json.loads(line)
                cid = record["custom_id"]
                resp = record.get("response", {})
                body = resp.get("body", {})
                choices = body.get("choices", [])
                text = ""
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                results_by_id[cid] = text

        if batch.error_file_id:
            file_ids.append(batch.error_file_id)
            error_content = self._client.files.content(batch.error_file_id)
            logger.warning("Batch 部分请求失败:\n%s", error_content.text[:2000])

        success_count = len(results_by_id)
        empty_count = expected_count - success_count
        logger.info("结果解析完成: %d/%d 成功, %d 空结果",
                     success_count, expected_count, empty_count)

        ordered: List[str] = []
        for idx in range(expected_count):
            ordered.append(results_by_id.get(str(idx), ""))
        return ordered, file_ids

    def _cleanup_remote_files(self, file_ids: List[str]) -> None:
        """删除远端已上传的文件，避免占用配额"""
        for fid in file_ids:
            try:
                self._client.files.delete(fid)
                logger.info("已删除远端文件: %s", fid)
            except Exception as e:
                logger.warning("删除远端文件 %s 失败: %s", fid, e)

    def __repr__(self) -> str:
        return (f"BatchTagClient(model={self.profile.model!r}, "
                f"base_url={self.profile.base_url!r})")


class BatchEmbeddingClient:
    """基于 Batch API 的批量 Embedding 客户端

    流程与 BatchTagClient 一致（上传 JSONL → 创建 Batch → 轮询 → 下载），
    区别在于 JSONL 每行为 embedding 请求，返回值为向量列表。

    Parameters
    ----------
    profile : LLMProfile
        LLM 配置（api_key / base_url / model）。
    batch_config : BatchConfig
        Batch API 运行参数（轮询间隔、超时）。
        注意：batch_config.endpoint 不会被使用，endpoint 固定为 /v1/embeddings。
    encoding_format : str
        Embedding 输出格式，默认 "float"。
    """

    ENDPOINT = "/v1/embeddings"

    def __init__(
        self,
        profile: LLMProfile,
        batch_config: BatchConfig = BatchConfig(),
        encoding_format: str = "float",
    ):
        self.profile = profile
        self.batch_config = batch_config
        self.encoding_format = encoding_format
        self._client = OpenAI(
            api_key=profile.api_key,
            base_url=profile.base_url,
            timeout=profile.timeout,
            max_retries=profile.max_retries,
        )

    # ----------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """通过 Batch API 批量获取 embedding，返回与输入顺序一致的向量列表

        对于失败或缺失的请求，返回空列表 []。
        """
        if not texts:
            return []

        remote_file_ids: List[str] = []
        jsonl_path = self._build_jsonl(texts)
        try:
            input_file_id = self._upload(jsonl_path)
            remote_file_ids.append(input_file_id)
            batch_id = self._create_batch(input_file_id)
            self._wait_for_completion(batch_id)
            results, extra_file_ids = self._download_results(
                batch_id, len(texts),
            )
            remote_file_ids.extend(extra_file_ids)
            return results
        finally:
            Path(jsonl_path).unlink(missing_ok=True)
            self._cleanup_remote_files(remote_file_ids)

    # ----------------------------------------------------------
    #  内部方法
    # ----------------------------------------------------------

    def _build_jsonl(self, texts: List[str]) -> str:
        """将文本列表写入 embedding 格式的临时 JSONL 文件"""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
        )
        for idx, text in enumerate(texts):
            record = {
                "custom_id": str(idx),
                "method": "POST",
                "url": self.ENDPOINT,
                "body": {
                    "model": self.profile.model,
                    "input": text,
                    "encoding_format": self.encoding_format,
                },
            }
            tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.close()
        logger.debug("Embedding JSONL 文件已生成: %s (%d 条)", tmp.name, len(texts))
        return tmp.name

    def _upload(self, jsonl_path: str) -> str:
        file_obj = self._client.files.create(
            file=Path(jsonl_path), purpose="batch",
        )
        logger.info("Embedding Batch 文件已上传, file_id=%s", file_obj.id)
        return file_obj.id

    def _create_batch(self, input_file_id: str) -> str:
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint=self.ENDPOINT,
            completion_window="24h",
        )
        logger.info("Embedding Batch 任务已创建, batch_id=%s", batch.id)
        return batch.id

    def _wait_for_completion(self, batch_id: str) -> None:
        elapsed = 0
        poll_count = 0
        while elapsed < self.batch_config.max_wait:
            batch = self._client.batches.retrieve(batch_id=batch_id)
            status = batch.status
            poll_count += 1

            completed = getattr(batch.request_counts, "completed", "?")
            failed = getattr(batch.request_counts, "failed", "?")
            total = getattr(batch.request_counts, "total", "?")
            logger.info(
                "[轮询 #%d] Embedding Batch %s | 状态: %s | 进度: %s/%s 完成, %s 失败 | 已等待 %ds",
                poll_count, batch_id, status, completed, total, failed, elapsed,
            )

            if status == "completed":
                logger.info("Embedding Batch %s 已完成 (共轮询 %d 次, 耗时 %ds)",
                            batch_id, poll_count, elapsed)
                return
            if status in ("failed", "expired", "cancelled"):
                raise RuntimeError(
                    f"Embedding Batch 任务 {status}: {batch.errors}"
                )

            time.sleep(self.batch_config.poll_interval)
            elapsed += self.batch_config.poll_interval

        raise TimeoutError(
            f"Embedding Batch 任务 {batch_id} 超时 ({self.batch_config.max_wait}s)"
        )

    def _download_results(
        self, batch_id: str, expected_count: int,
    ) -> tuple:
        """下载并解析 embedding 结果，按 custom_id 排序返回

        Returns (ordered_embeddings, file_ids_to_cleanup)
        """
        batch = self._client.batches.retrieve(batch_id=batch_id)
        file_ids: List[str] = []

        results_by_id: Dict[str, List[float]] = {}

        if batch.output_file_id:
            file_ids.append(batch.output_file_id)
            logger.info("下载 Embedding 输出文件: %s", batch.output_file_id)
            content = self._client.files.content(batch.output_file_id)
            for line in content.text.strip().split("\n"):
                if not line.strip():
                    continue
                record = json.loads(line)
                cid = record["custom_id"]
                resp = record.get("response", {})
                body = resp.get("body", {})
                data = body.get("data", [])
                embedding: List[float] = []
                if data:
                    embedding = data[0].get("embedding", [])
                results_by_id[cid] = embedding

        if batch.error_file_id:
            file_ids.append(batch.error_file_id)
            error_content = self._client.files.content(batch.error_file_id)
            logger.warning("Embedding Batch 部分请求失败:\n%s",
                           error_content.text[:2000])

        success_count = sum(1 for v in results_by_id.values() if v)
        empty_count = expected_count - success_count
        logger.info("Embedding 结果解析完成: %d/%d 成功, %d 空结果",
                     success_count, expected_count, empty_count)

        ordered: List[List[float]] = []
        for idx in range(expected_count):
            ordered.append(results_by_id.get(str(idx), []))
        return ordered, file_ids

    def _cleanup_remote_files(self, file_ids: List[str]) -> None:
        for fid in file_ids:
            try:
                self._client.files.delete(fid)
                logger.info("已删除远端文件: %s", fid)
            except Exception as e:
                logger.warning("删除远端文件 %s 失败: %s", fid, e)

    def __repr__(self) -> str:
        return (f"BatchEmbeddingClient(model={self.profile.model!r}, "
                f"base_url={self.profile.base_url!r})")
