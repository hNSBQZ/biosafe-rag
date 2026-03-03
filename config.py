"""
配置管理
========
集中管理所有外部依赖的配置（LLM API、Milvus、打标参数等）。
通过 load_config() 从 .env 文件加载，业务代码零硬编码。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


# ------------------------------------------------------------------
#  数据结构
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LLMProfile:
    """一个 LLM 接入点的完整配置"""
    api_key: str
    base_url: str
    model: str
    timeout: int = 60
    max_retries: int = 3


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus 连接与 Collection 配置

    uri 支持两种形式:
      - 本地文件路径 (Milvus Lite): "./milvus_lite.db"
      - 远程服务地址: "http://localhost:19530"
    """
    uri: str = "./milvus_lite.db"
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "biosafe_chunks"
    embedding_dim: int = 1024
    metric_type: str = "COSINE"


@dataclass(frozen=True)
class BatchConfig:
    """Batch API 调用参数"""
    poll_interval: int = 15
    max_wait: int = 7200
    endpoint: str = "/v1/chat/completions"


@dataclass(frozen=True)
class AppConfig:
    """应用全局配置"""
    llm_profiles: Dict[str, LLMProfile] = field(default_factory=dict)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)

    role_confidence_threshold: int = 3
    roles: Tuple[str, ...] = (
        "sop", "emergency", "regulation", "directory",
        "knowledge", "equipment", "reagent", "notice",
    )

    prompts: Dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------
#  Prompt 模板
# ------------------------------------------------------------------

DEFAULT_PROMPTS: Dict[str, str] = {
    "role_tagging": (
        "请判断以下文本片段最适合归入哪个知识类别。只返回一个类别名称。\n\n"
        "类别：sop / emergency / regulation / directory / "
        "knowledge / equipment / reagent / notice\n\n"
        "文本片段的层级路径：{heading_path}\n"
        "文本内容：\n{chunk_text}\n\n"
        "类别："
    ),
}


# ------------------------------------------------------------------
#  加载逻辑
# ------------------------------------------------------------------

_PROFILE_NAMES = ("tagging", "embedding", "chat")


def _load_env(env_path: str) -> Dict[str, str]:
    """解析 .env 文件为 dict（不污染 os.environ）"""
    env: Dict[str, str] = {}
    path = Path(env_path)
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def load_config(env_path: str = ".env") -> AppConfig:
    """从 .env 文件构建 AppConfig

    .env 中按前缀区分不同用途的 LLM 配置：
        TAGGING_API_KEY / TAGGING_BASE_URL / TAGGING_MODEL
        EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL
        CHAT_API_KEY / CHAT_BASE_URL / CHAT_MODEL

    缺省的 profile 会被跳过（不报错），使用时再检查。
    """
    env = _load_env(env_path)

    profiles: Dict[str, LLMProfile] = {}
    for name in _PROFILE_NAMES:
        prefix = name.upper()
        api_key = env.get(f"{prefix}_API_KEY", "")
        base_url = env.get(f"{prefix}_BASE_URL", "")
        model = env.get(f"{prefix}_MODEL", "")
        if api_key and base_url and model:
            profiles[name] = LLMProfile(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )

    milvus = MilvusConfig(
        uri=env.get("MILVUS_URI", "./milvus_lite.db"),
        host=env.get("MILVUS_HOST", "localhost"),
        port=int(env.get("MILVUS_PORT", "19530")),
        collection_name=env.get("MILVUS_COLLECTION", "biosafe_chunks"),
        embedding_dim=int(env.get("MILVUS_EMBEDDING_DIM", "1024")),
        metric_type=env.get("MILVUS_METRIC_TYPE", "COSINE"),
    )

    batch = BatchConfig(
        poll_interval=int(env.get("BATCH_POLL_INTERVAL", "15")),
        max_wait=int(env.get("BATCH_MAX_WAIT", "7200")),
        endpoint=env.get("BATCH_ENDPOINT", "/v1/chat/completions"),
    )

    prompts = dict(DEFAULT_PROMPTS)

    return AppConfig(
        llm_profiles=profiles,
        milvus=milvus,
        batch=batch,
        prompts=prompts,
    )
