# 对外函数 API 参考

本文档列出 `process_table.py` 和 `split_chunk.py` 对外暴露的公共函数，供上层模块（RAG pipeline、调试脚本等）调用。

> 以 `_` 开头的函数均为模块内部实现，不在此列出。

---

## process_table.py

### `parse_tables_from_text(text: str) -> list[dict]`

**定位**：核心解析函数，被 `split_chunk.py` 内部调用。

从 Markdown 纯文本中提取所有 HTML `<table>` 块，解析为结构化记录。

**参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | Markdown 文件的完整文本内容 |

**返回值**

`list[dict]`，每个 dict 代表一张表，结构如下：

```python
{
    "table_index": int,        # 表格在文档中的序号 (1-based)
    "table_name": str,         # 从表格前文推断的表名
    "columns": list[str],      # 列名列表
    "header_rows": int,        # 表头行数
    "is_transposed": bool,     # 是否为转置表 (表头在左侧)
    "row_count": int,          # 原始行数 (含表头)
    "record_count": int,       # 有效数据记录数
    "records": list[dict],     # 每条记录: {列名: 值, ...}
    "start_char": int,         # 在原文中的字符偏移 (内部字段，供 split_chunk 定位用)
}
```

**典型调用方**：`split_chunk.py` 的 `process_file()` 内部。

---

### `extract_tables_from_file(md_path: str | Path) -> list[dict]`

**定位**：独立使用的便捷函数，读取文件并仅返回表格数据。

读取指定 Markdown 文件，提取其中所有 HTML 表格，返回 JSON 可序列化的结构化数据（不含内部字段 `start_char`）。

**参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `md_path` | `str \| Path` | Markdown 文件路径 |

**返回值**

`list[dict]`，结构同 `parse_tables_from_text`，但去掉了 `start_char` 字段。

**使用示例**

```python
from process_table import extract_tables_from_file

tables = extract_tables_from_file("doc/名录.md")
for t in tables:
    print(f"表格: {t['table_name']}, {t['record_count']} 条记录")
    for rec in t["records"]:
        print(rec)
```

---

### `dump_debug_json(data: Any, out_path: str | Path, *, indent: int = 2) -> Path`

**定位**：调试辅助函数，将任意数据序列化到 JSON 文件。

自动创建父目录。返回实际写入的文件路径。

**参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | `Any` | 任意可 JSON 序列化的数据 |
| `out_path` | `str \| Path` | 输出文件路径 |
| `indent` | `int` | 缩进空格数，默认 `2` |

**返回值**

`Path` — 写入的文件绝对路径。

**使用示例**

```python
from process_table import extract_tables_from_file, dump_debug_json

tables = extract_tables_from_file("doc/名录.md")
dump_debug_json(tables, "debug/tables_output.json")
```

---

## split_chunk.py

### `process_file(filepath, ...) -> List[Block]`

**定位**：主入口函数，完成单个 Markdown 文件从解析到分块的全流程。

```python
def process_file(
    filepath: str,
    block_level_threshold: int = 2,
    chunk_max_tokens: int = 384,
    chunk_min_tokens: int = 128,
    block_max_tokens: int = 2000,
    block_min_tokens: int = 100,
) -> List[Block]
```

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `filepath` | `str` | — | Markdown 文件路径 |
| `block_level_threshold` | `int` | `2` | Block 边界的标题层级阈值（`<= 该值`的标题作为 Block 边界） |
| `chunk_max_tokens` | `int` | `384` | Chunk 最大 token 数 |
| `chunk_min_tokens` | `int` | `128` | Chunk 最小 token 数（过小则合并） |
| `block_max_tokens` | `int` | `2000` | Block 最大 token 数（过大则降级拆分） |
| `block_min_tokens` | `int` | `100` | Block 最小 token 数（过小则合并） |

**返回值**

`List[Block]`，每个 Block 含若干 Chunk，均已注入面包屑路径。内部会自动调用 `process_table.parse_tables_from_text` 处理文档中的 HTML 表格。

---

### `blocks_to_dicts(blocks: List[Block]) -> List[dict]`

**定位**：序列化辅助函数，将 Block 列表转为纯 dict 列表以便 JSON 导出。

**参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `blocks` | `List[Block]` | `process_file` 的返回值 |

**返回值**

`list[dict]`，结构如下：

```python
{
    "title": str,
    "source_file": str,
    "breadcrumb": list[str],
    "heading_level": int,
    "start_line": int,
    "end_line": int,
    "chunks": [
        {
            "title": str,
            "content": str,        # 含面包屑前缀的完整文本
            "start_line": int,
            "end_line": int,
            "heading_level": int,
            "token_estimate": int,
        },
        ...
    ]
}
```

---

### `format_display(blocks: List[Block]) -> str`

**定位**：调试辅助函数，生成人类可读的 Block-Chunk 树形结构概览文本。

**参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `blocks` | `List[Block]` | `process_file` 的返回值 |

**返回值**

`str` — 可直接 `print()` 的格式化文本。

---

## 典型调用流程

```python
from split_chunk import process_file, blocks_to_dicts, format_display
from process_table import extract_tables_from_file, dump_debug_json

# ── 完整分块流程 (文本 + 表格) ──
blocks = process_file("doc/应急预案.md")
print(format_display(blocks))                         # 终端预览
dump_debug_json(blocks_to_dicts(blocks), "debug/chunks.json")  # 序列化到文件

# ── 仅提取表格 ──
tables = extract_tables_from_file("doc/名录.md")
dump_debug_json(tables, "debug/tables.json")
```
