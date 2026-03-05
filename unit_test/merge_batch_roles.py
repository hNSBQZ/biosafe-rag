"""
将 Batch LLM 打标结果回填到 role_result.json。

用法示例:
    python unit_test/merge_batch_roles.py \
      --role-file role_result.json \
      --batch-file unit_test/file-batch_output-8ca20cd24ba64e57b5e1c8d4.jsonl \
      --out-file role_result_merged.json
"""

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple


VALID_ROLES = {
    "sop", "emergency", "regulation", "directory",
    "knowledge", "equipment", "reagent", "notice",
}


def parse_llm_role(raw: str) -> Optional[str]:
    """从 LLM 原始输出中提取合法 role。"""
    if not raw:
        return None
    cleaned = re.sub(r"[^a-z]", "", raw.strip().lower())
    if cleaned in VALID_ROLES:
        return cleaned
    low = raw.lower()
    for role in VALID_ROLES:
        if role in low:
            return role
    return None


def load_batch_results(batch_file: str) -> Dict[int, str]:
    """读取 batch 输出 jsonl，返回 custom_id(int) -> role(str)。"""
    by_custom_id: Dict[int, str] = {}
    with open(batch_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 第 {line_no} 行不是合法 JSON，已跳过")
                continue

            custom_id_raw = str(record.get("custom_id", "")).strip()
            if not custom_id_raw.isdigit():
                print(f"[WARN] 第 {line_no} 行 custom_id 非数字: {custom_id_raw!r}，已跳过")
                continue
            custom_id = int(custom_id_raw)

            body = (
                record.get("response", {})
                .get("body", {})
            )
            choices = body.get("choices", []) if isinstance(body, dict) else []
            content = ""
            if choices:
                content = (
                    choices[0]
                    .get("message", {})
                    .get("content", "")
                )
            role = parse_llm_role(content)
            if role:
                by_custom_id[custom_id] = role
            else:
                print(f"[WARN] 第 {line_no} 行无法解析 role: {content!r}")
    return by_custom_id


def pick_low_confidence_indices(
    records: List[dict],
    threshold: int,
    only_rule: bool,
) -> List[int]:
    """
    找出待 LLM 回填的条目在 role_result 中的下标。

    需与 chunk_processor 的 low_confidence 顺序一致：
    即按 role_result 的原始顺序扫描，收集低置信度项。
    """
    indices: List[int] = []
    for idx, rec in enumerate(records):
        conf = rec.get("role_confidence")
        if not isinstance(conf, int):
            continue
        if conf >= threshold:
            continue
        if only_rule and rec.get("tagged_by", "rule") != "rule":
            continue
        indices.append(idx)
    return indices


def merge_results(
    records: List[dict],
    low_indices: List[int],
    batch_roles: Dict[int, str],
) -> Tuple[int, int]:
    """执行回填，返回 (updated_count, missing_count)。"""
    updated = 0
    missing = 0
    for cid, role in batch_roles.items():
        if cid < 0 or cid >= len(low_indices):
            missing += 1
            print(f"[WARN] custom_id={cid} 超出低置信度列表范围，已跳过")
            continue
        rec_idx = low_indices[cid]
        rec = records[rec_idx]
        rec["role"] = role
        rec["tagged_by"] = "llm"
        updated += 1
    return updated, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="回填 Batch LLM 打标结果到 role_result.json")
    parser.add_argument("--role-file", default="role_result.json", help="原始 role_result.json 路径")
    parser.add_argument("--batch-file", required=True, help="Batch 输出 JSONL 路径")
    parser.add_argument("--out-file", default="", help="输出文件路径，默认覆盖 role_file")
    parser.add_argument("--threshold", type=int, default=3, help="低置信度阈值，默认 3")
    parser.add_argument(
        "--include-non-rule",
        action="store_true",
        help="默认仅回填 tagged_by=rule 的低置信项；开启后包含其他 tagged_by",
    )
    args = parser.parse_args()

    with open(args.role_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("role_file 内容应为 JSON 数组")

    only_rule = not args.include_non_rule
    low_indices = pick_low_confidence_indices(
        records=records,
        threshold=args.threshold,
        only_rule=only_rule,
    )
    batch_roles = load_batch_results(args.batch_file)

    updated, missing = merge_results(
        records=records,
        low_indices=low_indices,
        batch_roles=batch_roles,
    )
    out_file = args.out_file or args.role_file
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("=== 回填完成 ===")
    print(f"role_file 总条数: {len(records)}")
    print(f"低置信度候选数(role_confidence < {args.threshold}): {len(low_indices)}")
    print(f"batch 可解析结果数: {len(batch_roles)}")
    print(f"成功更新条数: {updated}")
    print(f"越界 custom_id 数: {missing}")
    print(f"输出文件: {out_file}")


if __name__ == "__main__":
    main()
