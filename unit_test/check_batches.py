"""
查看 Dashscope Batch 任务状态
===========================

支持两类能力：
1) list: 按过滤条件列出 Batch 任务
2) retrieve: 查看指定 batch_id 的详细状态

用法示例：
    python check_batches.py --limit 10
    python check_batches.py --status completed,expired --create-after 20250304000000
    python check_batches.py --ids batch_xxx,batch_yyy
    python check_batches.py --limit 5 --detail-top 3
"""

import argparse
import os
from typing import Dict, List, Optional

from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _load_dotenv(dotenv_path: str = ".env") -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not os.path.exists(dotenv_path):
        return env
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def _build_client(api_key_env: str, base_url: Optional[str]) -> OpenAI:
    dotenv = _load_dotenv(".env")

    api_key = (
        os.getenv(api_key_env)
        or dotenv.get(api_key_env)
        or os.getenv("DASHSCOPE_API_KEY")
        or dotenv.get("DASHSCOPE_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or dotenv.get("EMBEDDING_API_KEY")
        or os.getenv("TAGGING_API_KEY")
        or dotenv.get("TAGGING_API_KEY")
        or ""
    )
    if not api_key:
        raise ValueError(
            "未找到可用 API Key。请在环境变量或 .env 中设置："
            f"{api_key_env}（或 DASHSCOPE_API_KEY / EMBEDDING_API_KEY / TAGGING_API_KEY）"
        )

    resolved_base_url = (
        base_url
        or os.getenv("DASHSCOPE_BASE_URL")
        or dotenv.get("DASHSCOPE_BASE_URL")
        or os.getenv("EMBEDDING_BASE_URL")
        or dotenv.get("EMBEDDING_BASE_URL")
        or os.getenv("TAGGING_BASE_URL")
        or dotenv.get("TAGGING_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return OpenAI(api_key=api_key, base_url=resolved_base_url)


def _build_extra_query(args: argparse.Namespace) -> Dict[str, str]:
    extra: Dict[str, str] = {}
    if args.ds_name:
        extra["ds_name"] = args.ds_name
    if args.input_file_ids:
        extra["input_file_ids"] = args.input_file_ids
    if args.status:
        extra["status"] = args.status
    if args.create_after:
        extra["create_after"] = args.create_after
    if args.create_before:
        extra["create_before"] = args.create_before
    return extra


def _safe_counts(batch) -> str:
    counts = getattr(batch, "request_counts", None)
    if not counts:
        return "completed=? failed=? total=?"
    return (
        f"completed={getattr(counts, 'completed', '?')} "
        f"failed={getattr(counts, 'failed', '?')} "
        f"total={getattr(counts, 'total', '?')}"
    )


def _print_batch_detail(batch, idx: Optional[int] = None) -> None:
    prefix = f"[{idx}] " if idx is not None else ""
    print("-" * 90)
    print(f"{prefix}id={getattr(batch, 'id', '')}")
    print(
        f"status={getattr(batch, 'status', '')} "
        f"endpoint={getattr(batch, 'endpoint', '')} "
        f"completion_window={getattr(batch, 'completion_window', '')}"
    )
    print(_safe_counts(batch))
    print(f"input_file_id={getattr(batch, 'input_file_id', '')}")
    print(f"output_file_id={getattr(batch, 'output_file_id', '')}")
    print(f"error_file_id={getattr(batch, 'error_file_id', '')}")
    print(f"created_at={getattr(batch, 'created_at', '')}")
    print(f"in_progress_at={getattr(batch, 'in_progress_at', '')}")
    print(f"completed_at={getattr(batch, 'completed_at', '')}")
    errors = getattr(batch, "errors", None)
    if errors:
        print(f"errors={errors}")


def _list_batches(client: OpenAI, args: argparse.Namespace) -> List[str]:
    extra = _build_extra_query(args)
    kwargs = {
        "limit": args.limit,
    }
    if args.after:
        kwargs["after"] = args.after
    if extra:
        kwargs["extra_query"] = extra

    resp = client.batches.list(**kwargs)
    items = getattr(resp, "data", []) or []
    print(f"\n共返回 {len(items)} 个任务")
    print("=" * 90)
    for i, batch in enumerate(items):
        _print_batch_detail(batch, i)
    return [getattr(b, "id", "") for b in items if getattr(b, "id", "")]


def _retrieve_batches(client: OpenAI, batch_ids: List[str]) -> None:
    if not batch_ids:
        return
    print("\n\n指定任务详情")
    print("=" * 90)
    for i, bid in enumerate(batch_ids):
        bid = bid.strip()
        if not bid:
            continue
        try:
            batch = client.batches.retrieve(bid)
            _print_batch_detail(batch, i)
        except Exception as e:
            print("-" * 90)
            print(f"[{i}] id={bid}")
            print(f"retrieve 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="查看 Dashscope Batch 任务状态")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY",
                        help="API Key 环境变量名，默认 DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
                        help=f"Base URL，默认 {DEFAULT_BASE_URL}")

    # list 参数
    parser.add_argument("--after", default="", help="分页游标 after=batch_xxx")
    parser.add_argument("--limit", type=int, default=10, help="返回任务数，默认 10")
    parser.add_argument("--ds-name", default="", help="任务名称过滤")
    parser.add_argument("--input-file-ids", default="", help="输入文件过滤，逗号分隔")
    parser.add_argument("--status", default="", help="状态过滤，如 completed,expired")
    parser.add_argument("--create-after", default="", help="创建时间下限，格式 yyyyMMddHHmmss")
    parser.add_argument("--create-before", default="", help="创建时间上限，格式 yyyyMMddHHmmss")

    # retrieve 参数
    parser.add_argument("--ids", default="",
                        help="指定 batch_id 列表，逗号分隔，如 batch_xxx,batch_yyy")
    parser.add_argument("--detail-top", type=int, default=0,
                        help="从 list 结果中自动取前 N 个任务做 retrieve 详情")

    args = parser.parse_args()

    client = _build_client(args.api_key_env, args.base_url)

    listed_ids = _list_batches(client, args)

    manual_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    auto_ids = listed_ids[:max(args.detail_top, 0)]
    merged_ids = []
    seen = set()
    for bid in manual_ids + auto_ids:
        if bid and bid not in seen:
            seen.add(bid)
            merged_ids.append(bid)

    _retrieve_batches(client, merged_ids)


if __name__ == "__main__":
    main()

