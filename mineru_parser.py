"""
MineRU 离线文档解析客户端
=========================
封装 MineRU OpenAPI，提供：
  - submit_task()   提交文档解析任务（URL 或文件上传）
  - query_task()    查询任务状态与结果
  - wait_for_task() 轮询等待任务完成
"""

import time
import logging
from pathlib import Path
from typing import Optional

import requests

from config import _load_env

logger = logging.getLogger(__name__)

MINERU_BASE_URL = "https://mineru.net/api/v4"


def _get_token(env_path: str = ".env") -> str:
    env = _load_env(env_path)
    token = env.get("MINERU_API_TOKEN", "")
    if not token:
        raise ValueError("MINERU_API_TOKEN 未配置，请在 .env 中填写")
    return token


def _headers(token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


# ------------------------------------------------------------------
#  提交任务
# ------------------------------------------------------------------

def submit_task(
    file_url: str,
    *,
    model_version: str = "vlm",
    enable_table: bool = True,
    enable_formula: bool = True,
    language: str = "ch",
    env_path: str = ".env",
    timeout: int = 30,
) -> dict:
    """提交一个文档解析任务。

    Args:
        file_url:       待解析文档的公网可访问 URL。
        model_version:  模型版本，"vlm"（高精度）或 "doclayout"（快速）。
        enable_table:   是否开启表格识别。
        enable_formula: 是否开启公式识别。
        language:       文档语言，"ch" / "en" 等。
        env_path:       .env 文件路径。
        timeout:        请求超时秒数。

    Returns:
        API 返回的完整 JSON（含 task_id 等信息）。
    """
    token = _get_token(env_path)
    url = f"{MINERU_BASE_URL}/extract/task"
    payload = {
        "url": file_url,
        "model_version": model_version,
        "enable_table": enable_table,
        "enable_formula": enable_formula,
        "language": language,
        "is_ocr": True,
    }

    resp = requests.post(url, headers=_headers(token), json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    logger.info("任务已提交: %s", result)
    return result


def submit_task_by_file(
    file_path: str,
    *,
    model_version: str = "vlm",
    enable_table: bool = True,
    enable_formula: bool = True,
    language: str = "ch",
    env_path: str = ".env",
    timeout: int = 60,
) -> dict:
    """通过本地文件上传提交解析任务。

    Args:
        file_path:      本地文件路径。
        model_version:  模型版本。
        enable_table:   是否开启表格识别。
        enable_formula: 是否开启公式识别。
        language:       文档语言。
        env_path:       .env 文件路径。
        timeout:        请求超时秒数。

    Returns:
        API 返回的完整 JSON。
    """
    token = _get_token(env_path)
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    url = f"{MINERU_BASE_URL}/extract/task/file"
    data = {
        "model_version": model_version,
        "enable_table": str(enable_table).lower(),
        "enable_formula": str(enable_formula).lower(),
        "language": language,
        "is_ocr": True,
    }

    with open(p, "rb") as f:
        files = {"file": (p.name, f, "application/octet-stream")}
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)

    resp.raise_for_status()
    result = resp.json()
    logger.info("文件上传任务已提交: %s", result)
    return result


# ------------------------------------------------------------------
#  查询任务状态
# ------------------------------------------------------------------

def query_task(task_id: str, *, env_path: str = ".env", timeout: int = 15) -> dict:
    """查询指定任务的状态与结果。

    Args:
        task_id: 提交任务时返回的 task_id。
        env_path: .env 文件路径。
        timeout:  请求超时秒数。

    Returns:
        API 返回的完整 JSON，包含 state / progress / data 等字段。
    """
    token = _get_token(env_path)
    url = f"{MINERU_BASE_URL}/extract/task/{task_id}"

    resp = requests.get(url, headers=_headers(token), timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    logger.debug("任务 %s 状态: %s", task_id, result)
    return result


# ------------------------------------------------------------------
#  轮询等待
# ------------------------------------------------------------------

def wait_for_task(
    task_id: str,
    *,
    poll_interval: int = 10,
    max_wait: int = 600,
    env_path: str = ".env",
) -> dict:
    """轮询等待任务完成。

    Args:
        task_id:       任务 ID。
        poll_interval: 轮询间隔（秒）。
        max_wait:      最大等待时间（秒），超时抛 TimeoutError。
        env_path:      .env 文件路径。

    Returns:
        任务完成后的完整 JSON。
    """
    elapsed = 0
    while elapsed < max_wait:
        result = query_task(task_id, env_path=env_path)
        data = result.get("data", {})
        state = data.get("state", "unknown")

        if state == "done":
            logger.info("任务 %s 已完成", task_id)
            return result
        if state == "failed":
            raise RuntimeError(f"任务 {task_id} 失败: {result}")

        logger.info("任务 %s 状态: %s，%d 秒后重试…", task_id, state, poll_interval)
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"任务 {task_id} 在 {max_wait}s 内未完成")


# ------------------------------------------------------------------
#  CLI 快捷入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="MineRU 文档解析工具")
    sub = parser.add_subparsers(dest="cmd")

    p_submit = sub.add_parser("submit", help="提交解析任务（URL）")
    p_submit.add_argument("url", help="文档的公网 URL")
    p_submit.add_argument("--model", default="vlm", choices=["vlm", "doclayout"])

    p_upload = sub.add_parser("upload", help="上传本地文件并解析")
    p_upload.add_argument("file", help="本地文件路径")
    p_upload.add_argument("--model", default="vlm", choices=["vlm", "doclayout"])

    p_query = sub.add_parser("query", help="查询任务状态")
    p_query.add_argument("task_id", help="任务 ID")

    p_wait = sub.add_parser("wait", help="等待任务完成")
    p_wait.add_argument("task_id", help="任务 ID")
    p_wait.add_argument("--interval", type=int, default=10)
    p_wait.add_argument("--max-wait", type=int, default=600)

    args = parser.parse_args()

    if args.cmd == "submit":
        print(submit_task(args.url, model_version=args.model))
    elif args.cmd == "upload":
        print(submit_task_by_file(args.file, model_version=args.model))
    elif args.cmd == "query":
        print(query_task(args.task_id))
    elif args.cmd == "wait":
        print(wait_for_task(args.task_id, poll_interval=args.interval, max_wait=args.max_wait))
    else:
        parser.print_help()
