"""
Block-Chunk 文档分块器 (方案一: 自顶向下)
==========================================
适用于生物安全领域的 Markdown 文档。

整体流程:
    原始 .md → 逐行清洗 → 标题/层级识别 → Block 切分 → Block 内 Chunk 切分 → 大小约束修正
    同时提取 HTML 表格, 每条表格记录作为独立 Chunk, 受相同大小约束

设计原则:
    - Block 由"高层级标题"划分, 代表一个语义主题 (如 "BSL-3实验室", "应急处理程序")
    - Chunk 由 Block 内的"子标题/编号条目"划分, 代表主题下的具体条款或步骤
    - 表格记录各自成为独立 Chunk, 过小则合并相邻记录, 过大则按段落拆分
    - 每个 Block/Chunk 携带 breadcrumb(面包屑路径), 方便后续打 role 和溯源

用法:
    python split_chunk.py [file1.md file2.md ...]
    不带参数则处理当前目录下所有 .md 文件
"""

import re
import json
import glob
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


# ============================================================
#  配置常量
# ============================================================

# Block 切分阈值: heading level <= 此值的标题作为 Block 边界
# level 1 = 顶级章节 (如 "# 3 风险评估", "# 一、目的")
# level 2 = 二级节   (如 "# 6.1 BSL-1 实验室", "（一）病原微生物...")
BLOCK_LEVEL_THRESHOLD = 2

# Chunk 大小约束 (以估算 token 数计)
CHUNK_MAX_TOKENS = 384   # 超过此值尝试拆分
CHUNK_MIN_TOKENS = 128   # 低于此值与相邻 chunk 合并

# Block 大小约束 (以估算 token 数计)
BLOCK_MAX_TOKENS = 2000  # 超过此值按子标题动态拆分
BLOCK_MIN_TOKENS = 100   # 低于此值与相邻 block 合并

# 空壳 block 同级合并的标题长度上限 (字符数)
# 短标题 (如 "二．操作步骤") 视为章节容器, 可作为伪父级合并到下一个同级 block
# 长标题 (如 "3.2实验室风险评估和风险控制活动的复杂程度...") 是完整条款, 不作父级合并
SHELL_TITLE_MAX_LEN = 30

# 编号项被识别为 chunk 边界的最小长度 (字符数)
# 短列表项 (如 "1. DNA 模板") 不会被切成独立 chunk
NUMBERED_ITEM_MIN_LENGTH = 50

# 页眉页脚等噪声的正则模式
NOISE_PATTERNS = [
    r'^Tel[:：].*$',
    r'^www\..*\.com.*$',
    r'^\S+@\S+\.com.*$',
    r'^德泰生物.*$',
    r'^DETAIBIO\s*$',
    r'^活性蛋白整体方案.*$',
    r'^Active Protein Solutions\s*$',
]


# ============================================================
#  数据结构
# ============================================================

@dataclass
class ParsedLine:
    """文档中每一行的解析结果"""
    line_no: int          # 原始行号 (1-based)
    raw: str              # 原始文本
    cleaned: str          # 清洗后文本
    heading_level: int    # 标题层级, 0 表示非标题
    heading_title: str    # 标题文本 (仅标题行有值)
    is_noise: bool        # 是否为噪声行
    is_table: bool        # 是否为表格行
    is_empty: bool        # 是否为空行


@dataclass
class Chunk:
    """Block 内的最小检索单元"""
    title: str            # chunk 的标题或首行摘要
    content: str          # 完整文本内容
    start_line: int       # 在源文件中的起始行
    end_line: int         # 在源文件中的结束行
    heading_level: int    # 触发此 chunk 的标题层级
    token_estimate: int = 0


@dataclass
class Block:
    """语义主题单元, 包含若干 Chunk"""
    title: str
    chunks: List[Chunk] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    heading_level: int = 0
    source_file: str = ""
    # 面包屑: 从文档根到当前 block 的标题路径
    breadcrumb: List[str] = field(default_factory=list)


# ============================================================
#  Step 1: 预处理 — 噪声清洗
# ============================================================

def clean_latex_fragment(tex: str) -> str:
    """将 OCR 产生的 LaTeX 残留片段还原为纯文本
    例: '$0 . 5 \\mathsf { m l }$' → '0.5ml'
    """
    # 去除常见 LaTeX 命令, 保留内容
    tex = re.sub(
        r'\\(?:mathsf|mathrm|textrm|mathord|scriptscriptstyle|substack)\s*',
        '', tex
    )
    tex = re.sub(r'\\(?:sim|~)', '~', tex)
    # 清理花括号
    tex = tex.replace('{', '').replace('}', '')
    # 去除剩余反斜杠命令
    tex = re.sub(r'\\[a-zA-Z]+', '', tex)
    # 合并多余空格 (OCR 常在数字间插入空格, 如 "0 . 5" → "0.5")
    tex = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', tex)
    tex = re.sub(r'\s{2,}', ' ', tex)
    return tex.strip()


def clean_line(line: str) -> str:
    """清洗单行文本: 处理 LaTeX 残留"""
    # 将 $...$ 包裹的 LaTeX 片段还原
    result = re.sub(
        r'\$([^$]+)\$',
        lambda m: clean_latex_fragment(m.group(1)),
        line
    )
    return result.strip()


def is_noise_line(line: str) -> bool:
    """判断是否为页眉/页脚等噪声行"""
    stripped = line.strip()
    if not stripped:
        return False
    return any(re.match(p, stripped, re.IGNORECASE) for p in NOISE_PATTERNS)


def is_table_line(line: str) -> bool:
    """判断是否为 HTML 表格行 (以 <table 开头)"""
    return bool(re.match(r'\s*<table', line.strip(), re.IGNORECASE))


# ============================================================
#  Step 2: 行级解析 — 标题识别与层级判定
# ============================================================

def _numbering_depth(text: str) -> int:
    """从文本开头提取点分编号的深度
    "6.3.1 xxx" → 3,  "3.1 xxx" → 2,  "一、xxx" → 1
    """
    # 点分数字编号 (支持全角/半角点号)
    m = re.match(r'^(\d+(?:[.．]\d+)*)', text)
    if m:
        numbering = m.group(1)
        dots = len(re.findall(r'[.．]', numbering))
        return dots + 1
    # 中文一级序号
    if re.match(r'^[一二三四五六七八九十]+[、．.]', text):
        return 1
    # 带括号的中文序号
    if re.match(r'^[（(][一二三四五六七八九十]+[)）]', text):
        return 2
    return 0


def detect_heading(line: str) -> Tuple[int, str]:
    """检测一行是否为标题, 返回 (level, title_text)
    level=0 表示非标题

    识别策略 (按优先级):
    1. Markdown # 标记 — 再结合内部编号修正层级
    2. 中文一级序号: 一、 二、 三、
    3. 点分数字编号 (至少含一个点): 3.1 / 6.3.2.1
    4. 带括号中文序号: （一） （二）
    """
    stripped = line.strip()
    if not stripped:
        return 0, ""

    # --- Markdown 标题 ---
    md_match = re.match(r'^(#{1,6})\s+(.*)', stripped)
    if md_match:
        hashes = len(md_match.group(1))
        title = md_match.group(2).strip()
        # 用内部编号修正层级 (因为很多文档所有标题都只用单个 #)
        inner = _numbering_depth(title)
        level = max(hashes, inner) if inner > 0 else hashes
        return min(level, 5), title

    # --- 中文一级序号 (无 # 前缀) ---
    if re.match(r'^[一二三四五六七八九十]+[、．.]\s*', stripped):
        return 1, stripped

    # --- 点分数字编号 (无 # 前缀, 要求至少 N.M 格式) ---
    num_match = re.match(r'^(\d+)([.．]\d+)+\s*(.*)', stripped)
    if num_match:
        first_num = int(num_match.group(1))
        # 过滤误判: 首段为 0 (如 "0.25%") 或 >= 1000 (如 "2016.6") 不是章节号
        if 1 <= first_num < 1000:
            rest = num_match.group(3)
            # 编号后面应跟中文/字母 (标题), 而非符号 (如 %, $ 等)
            if not rest or re.match(r'[\u4e00-\u9fffA-Za-z(（]', rest):
                numbering_part = re.match(r'^(\d+(?:[.．]\d+)+)', stripped).group(1)
                depth = len(re.findall(r'[.．]', numbering_part)) + 1
                return min(depth, 5), stripped

    # --- 带括号中文序号 ---
    if re.match(r'^[（(][一二三四五六七八九十]+[)）][、.]?\s*', stripped):
        return 2, stripped

    return 0, ""


def parse_document(lines: List[str]) -> List[ParsedLine]:
    """将文档的原始行列表解析为结构化的 ParsedLine 列表"""
    parsed = []
    for i, line in enumerate(lines):
        raw = line.rstrip('\n')
        is_empty = not raw.strip()
        noise = is_noise_line(raw)
        table = is_table_line(raw)

        if noise or table or is_empty:
            cleaned = ''
            heading_level, heading_title = 0, ""
        else:
            cleaned = clean_line(raw)
            heading_level, heading_title = detect_heading(cleaned)

        parsed.append(ParsedLine(
            line_no=i + 1,
            raw=raw,
            cleaned=cleaned,
            heading_level=heading_level,
            heading_title=heading_title,
            is_noise=noise,
            is_table=table,
            is_empty=is_empty,
        ))
    return parsed


# ============================================================
#  Step 3: Block 切分
# ============================================================

def split_into_blocks(
    parsed_lines: List[ParsedLine],
    source_file: str = "",
    block_level_threshold: int = BLOCK_LEVEL_THRESHOLD,
) -> List[Block]:
    """按高层级标题切分 Block

    所有 heading_level <= block_level_threshold 的标题行作为 Block 边界。
    每个 Block 从其标题行开始, 到下一个 Block 边界 (或文档末尾) 结束。
    """
    # 收集所有 block 边界点
    boundaries = []  # [(line_index, level, title)]
    for i, pl in enumerate(parsed_lines):
        if pl.heading_level > 0 and pl.heading_level <= block_level_threshold:
            boundaries.append((i, pl.heading_level, pl.heading_title))

    # 没有找到任何标题 → 整篇文档作为一个 block
    if not boundaries:
        boundaries = [(0, 1, source_file or "全文")]

    # 如果第一个 block 不从第 0 行开始, 为前言部分补一个 block
    if boundaries[0][0] > 0:
        preamble_title = _find_preamble_title(parsed_lines, boundaries[0][0])
        if preamble_title:
            boundaries.insert(0, (0, 0, preamble_title))

    # 构建 Block 对象
    blocks = []
    breadcrumb_stack: List[Tuple[int, str]] = []  # (level, title) 栈

    for b_idx, (start_idx, level, title) in enumerate(boundaries):
        end_idx = (boundaries[b_idx + 1][0]
                   if b_idx + 1 < len(boundaries) else len(parsed_lines))

        # 维护面包屑: 弹出同级或更深的条目, 压入当前
        while breadcrumb_stack and breadcrumb_stack[-1][0] >= level:
            breadcrumb_stack.pop()
        breadcrumb_stack.append((level, title))

        block_lines = parsed_lines[start_idx:end_idx]

        block = Block(
            title=title,
            start_line=parsed_lines[start_idx].line_no,
            end_line=parsed_lines[min(end_idx, len(parsed_lines)) - 1].line_no,
            heading_level=level,
            source_file=source_file,
            breadcrumb=[t for _, t in breadcrumb_stack],
        )
        blocks.append((block, block_lines))

    return blocks


def _find_preamble_title(parsed_lines: List[ParsedLine], before_idx: int) -> str:
    """在文档开头 (第一个 block 标题之前) 寻找一个合适的前言标题"""
    for pl in parsed_lines[:before_idx]:
        if not pl.is_noise and not pl.is_empty and not pl.is_table and pl.cleaned:
            return pl.cleaned[:60]
    return ""


def _merge_shell_blocks(
    block_tuples: List[Tuple["Block", List[ParsedLine]]],
) -> List[Tuple["Block", List[ParsedLine]]]:
    """合并空壳 Block: 仅当空壳是下一个 block 的父级时才合并

    典型场景: "# 六、应急处理程序"(level 1) 后面紧跟 "# （一）..."(level 2),
    前者是后者的父标题, 合并后前者的标题成为后者面包屑的一部分。

    同级兄弟空壳 (如 "3.2" 和 "3.4" 都是 level 2) 不在此处合并,
    留给 _merge_tiny_blocks 按 token 数处理。
    """
    if len(block_tuples) <= 1:
        return block_tuples

    result = []
    pending_shell = None  # 待合并的空壳 block

    def _is_shell(lines: List[ParsedLine]) -> bool:
        content_lines = [
            pl for pl in lines
            if not pl.is_noise and not pl.is_table and not pl.is_empty
            and pl.heading_level == 0
        ]
        return not content_lines and len(lines) <= 3

    def _flush_pending():
        nonlocal pending_shell
        if pending_shell is not None:
            result.append(pending_shell)
            pending_shell = None

    for block, lines in block_tuples:
        if _is_shell(lines):
            if pending_shell is None:
                pending_shell = (block, lines)
            else:
                prev_block, prev_lines = pending_shell
                if block.heading_level > prev_block.heading_level:
                    # 更深层级: 是 pending 的子标题, 合并 (如 "六" + "1. 总则")
                    pending_shell = (prev_block, prev_lines + lines)
                else:
                    # 同级或更浅: 不是父子关系, 先输出 pending 再重新开始
                    _flush_pending()
                    pending_shell = (block, lines)
        else:
            if pending_shell is not None:
                shell_block, shell_lines = pending_shell
                is_parent = shell_block.heading_level < block.heading_level
                is_short_same_level = (
                    shell_block.heading_level == block.heading_level
                    and len(shell_block.title) <= SHELL_TITLE_MAX_LEN
                )
                if is_parent or is_short_same_level:
                    # 空壳是父级, 或同级短标题 (章节容器) → 合并, 更新面包屑
                    merged_lines = shell_lines + lines
                    if shell_block.title not in block.breadcrumb:
                        block.breadcrumb = shell_block.breadcrumb + [
                            t for t in block.breadcrumb
                            if t not in shell_block.breadcrumb
                        ]
                    block.start_line = shell_block.start_line
                    result.append((block, merged_lines))
                else:
                    # 同级长标题 (完整条款) 或更深 → 不是父子, 各自独立
                    result.append(pending_shell)
                    result.append((block, lines))
                pending_shell = None
            else:
                result.append((block, lines))

    if pending_shell is not None:
        result.append(pending_shell)

    return result


# ============================================================
#  Step 3.6: 动态拆分超大 Block
# ============================================================

def _split_oversized_blocks(
    block_tuples: List[Tuple["Block", List[ParsedLine]]],
    max_tokens: int = BLOCK_MAX_TOKENS,
) -> List[Tuple["Block", List[ParsedLine]]]:
    """动态拆分超大 Block

    当 Block 的总 token 超过 max_tokens 时, 找到 Block 内最浅的子标题级别,
    以该级别为边界将 Block 拆分为若干子 Block。
    迭代执行直到所有 Block 都在限制内 (或已无更深子标题可拆)。
    """
    changed = True
    while changed:
        changed = False
        new_tuples: List[Tuple["Block", List[ParsedLine]]] = []
        for block, lines in block_tuples:
            token_est = _estimate_tokens(_assemble_content(lines))
            if token_est <= max_tokens:
                new_tuples.append((block, lines))
                continue

            sub = _try_split_block_at_subheadings(block, lines)
            if sub is not None:
                new_tuples.extend(sub)
                changed = True
            else:
                new_tuples.append((block, lines))
        block_tuples = new_tuples
    return block_tuples


def _try_split_block_at_subheadings(
    block: "Block", lines: List[ParsedLine],
) -> Optional[List[Tuple["Block", List[ParsedLine]]]]:
    """尝试在 Block 内部的子标题处拆分

    找到 block 内 (跳过首行) 最浅的子标题层级, 以此为边界拆分。
    返回新的 (Block, lines) 列表; 无法拆分时返回 None。
    """
    min_sub_level: Optional[int] = None
    for i, pl in enumerate(lines):
        if i == 0:
            continue
        if pl.heading_level > 0 and pl.heading_level > block.heading_level:
            if min_sub_level is None or pl.heading_level < min_sub_level:
                min_sub_level = pl.heading_level

    if min_sub_level is None:
        return None

    boundaries: List[Tuple[int, int, str]] = []
    for i, pl in enumerate(lines):
        if i == 0:
            continue
        if (pl.heading_level > 0
                and block.heading_level < pl.heading_level <= min_sub_level):
            boundaries.append((i, pl.heading_level, pl.heading_title))

    if not boundaries:
        return None

    if boundaries[0][0] > 0:
        boundaries.insert(0, (0, block.heading_level, block.title))

    result: List[Tuple["Block", List[ParsedLine]]] = []
    for b_idx, (start, level, title) in enumerate(boundaries):
        end = (boundaries[b_idx + 1][0]
               if b_idx + 1 < len(boundaries) else len(lines))
        sub_lines = lines[start:end]

        sub_bc = (list(block.breadcrumb) if start == 0
                  else list(block.breadcrumb) + [title])

        sub_block = Block(
            title=title,
            start_line=sub_lines[0].line_no,
            end_line=sub_lines[-1].line_no,
            heading_level=level,
            source_file=block.source_file,
            breadcrumb=sub_bc,
        )
        result.append((sub_block, sub_lines))

    return result if len(result) > 1 else None


# ============================================================
#  Step 3.7: 合并过小 Block
# ============================================================

def _merge_tiny_blocks(
    block_tuples: List[Tuple["Block", List[ParsedLine]]],
    min_tokens: int = BLOCK_MIN_TOKENS,
) -> List[Tuple["Block", List[ParsedLine]]]:
    """合并过小的 Block

    采用累积器模式: 当前累积 block 的 token 不足 min_tokens 时, 向后并入下一个 block。
    不跨顶级章节合并 (breadcrumb[0] 不同时不合并)。
    末尾的过小 block 向前并入 (同章内)。
    """
    if len(block_tuples) <= 1:
        return block_tuples

    def _chapter_root(blk: "Block") -> str:
        return blk.breadcrumb[0] if blk.breadcrumb else ""

    result: List[Tuple["Block", List[ParsedLine]]] = []
    acc_block: Optional["Block"] = None
    acc_lines: List[ParsedLine] = []
    acc_tokens = 0

    for block, lines in block_tuples:
        tokens = _estimate_tokens(_assemble_content(lines))

        if acc_block is None:
            acc_block, acc_lines, acc_tokens = block, lines, tokens
            continue

        if acc_tokens < min_tokens:
            if _chapter_root(acc_block) == _chapter_root(block):
                # 同章内合并
                merged_lines = acc_lines + lines
                merged_tokens = _estimate_tokens(
                    _assemble_content(merged_lines))
                block.start_line = acc_block.start_line
                acc_block, acc_lines, acc_tokens = (
                    block, merged_lines, merged_tokens)
            else:
                # 跨章: 不合并, 输出当前累积, 开始新累积
                result.append((acc_block, acc_lines))
                acc_block, acc_lines, acc_tokens = block, lines, tokens
        else:
            result.append((acc_block, acc_lines))
            acc_block, acc_lines, acc_tokens = block, lines, tokens

    if acc_block is not None:
        if acc_tokens < min_tokens and result:
            prev_block, prev_lines = result[-1]
            if _chapter_root(prev_block) == _chapter_root(acc_block):
                prev_block.end_line = acc_block.end_line
                result[-1] = (prev_block, prev_lines + acc_lines)
            else:
                result.append((acc_block, acc_lines))
        else:
            result.append((acc_block, acc_lines))

    # 再扫一遍: 把仍然过小的 block 向前合并 (同章内)
    i = len(result) - 1
    while i > 0:
        blk, lns = result[i]
        tok = _estimate_tokens(_assemble_content(lns))
        if tok < min_tokens:
            prev_blk, prev_lns = result[i - 1]
            if _chapter_root(prev_blk) == _chapter_root(blk):
                prev_blk.end_line = blk.end_line
                result[i - 1] = (prev_blk, prev_lns + lns)
                result.pop(i)
        i -= 1

    return result


# ============================================================
#  Step 4: Block 内 Chunk 切分
# ============================================================

def split_block_into_chunks(
    block_lines: List[ParsedLine],
    max_tokens: int = CHUNK_MAX_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
) -> List[Chunk]:
    """在一个 Block 内部, 按子标题和编号条目切分 Chunk

    Chunk 边界来源:
    1. 子标题 (heading_level > 0 且不是 block 自身的第一行标题)
    2. "实质性"编号条目 (内容足够长, 避免把短列表项各自切开)
    """
    if not block_lines:
        return []

    # 收集 chunk 切分点
    chunk_starts = []  # [(index_in_block, level, title)]
    for i, pl in enumerate(block_lines):
        if pl.is_noise or pl.is_table or pl.is_empty:
            continue

        # 子标题 → chunk 边界 (跳过 block 自身的首行标题)
        if pl.heading_level > 0 and i > 0:
            chunk_starts.append((i, pl.heading_level, pl.heading_title))
            continue

        # 实质性编号条目 → chunk 边界
        if i > 0 and _is_substantial_numbered_item(pl.cleaned):
            chunk_starts.append((i, 99, pl.cleaned[:60]))

    # 无子切分点 → 整个 block 作为一个 chunk
    if not chunk_starts:
        content = _assemble_content(block_lines)
        if not content.strip():
            return []
        return [Chunk(
            title=(block_lines[0].heading_title
                   or block_lines[0].cleaned[:60]),
            content=content,
            start_line=block_lines[0].line_no,
            end_line=block_lines[-1].line_no,
            heading_level=block_lines[0].heading_level,
            token_estimate=_estimate_tokens(content),
        )]

    # 如果第一个 chunk 不从 block 起始位置开始, 补一个
    if chunk_starts[0][0] > 0:
        first = block_lines[0]
        chunk_starts.insert(0, (
            0,
            first.heading_level or 99,
            first.heading_title or first.cleaned[:60],
        ))

    # 构建原始 chunk 列表
    raw_chunks = []
    for c_idx, (start, level, title) in enumerate(chunk_starts):
        end = (chunk_starts[c_idx + 1][0]
               if c_idx + 1 < len(chunk_starts) else len(block_lines))
        lines_slice = block_lines[start:end]
        content = _assemble_content(lines_slice)
        if content.strip():
            raw_chunks.append(Chunk(
                title=title,
                content=content,
                start_line=lines_slice[0].line_no,
                end_line=lines_slice[-1].line_no,
                heading_level=level,
                token_estimate=_estimate_tokens(content),
            ))

    # 大小约束: 合并过小 → 拆分过大 → 再合并 (拆分可能产生碎片)
    chunks = _merge_small_chunks(raw_chunks, min_tokens)
    chunks = _split_large_chunks(chunks, max_tokens)
    chunks = _merge_small_chunks(chunks, min_tokens)
    return chunks


def _is_substantial_numbered_item(text: str) -> bool:
    """判断一行是否是"实质性"编号条目, 即内容足够长, 值得作为独立 chunk

    目的: 避免把短列表 (如 "1. DNA 模板") 拆成单独 chunk,
    只有内容较长的编号段落 (如 "1、实验室如果发生一般病原微生物泼溅...") 才拆分
    """
    m = re.match(r'^(\d+)[、.．]\s*(.+)', text.strip())
    if not m:
        return False
    return len(m.group(2)) >= NUMBERED_ITEM_MIN_LENGTH


def _assemble_content(lines: List[ParsedLine]) -> str:
    """将 ParsedLine 列表组装为干净的文本, 跳过噪声行和表格行 (表格由独立 chunk 处理)"""
    parts = []
    for pl in lines:
        if pl.is_noise:
            continue
        if pl.is_table:
            continue
        if pl.is_empty:
            if parts and parts[-1] != '':
                parts.append('')
            continue
        parts.append(pl.cleaned)

    text = '\n'.join(parts)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数 (中英混合文本)
    中文字符 ≈ 1.5 token, 英文单词 ≈ 1 token
    """
    cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = len(re.findall(r'[a-zA-Z]+', text))
    num = len(re.findall(r'\d+', text))
    return int(cn * 1.5 + en + num * 0.5) + 1


# ============================================================
#  Step 5: 大小约束 — 合并与拆分
# ============================================================

def _merge_small_chunks(chunks: List[Chunk], min_tokens: int) -> List[Chunk]:
    """将过小的 chunk 向后合并到相邻 chunk"""
    if len(chunks) <= 1:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        # 如果当前 chunk 太小, 且后面还有 chunk, 就合并
        while (i + 1 < len(chunks)
               and current.token_estimate < min_tokens):
            nxt = chunks[i + 1]
            current = Chunk(
                title=current.title,
                content=current.content + '\n\n' + nxt.content,
                start_line=current.start_line,
                end_line=nxt.end_line,
                heading_level=current.heading_level,
                token_estimate=_estimate_tokens(
                    current.content + '\n\n' + nxt.content),
            )
            i += 1
        merged.append(current)
        i += 1
    return merged


def _split_large_chunks(chunks: List[Chunk], max_tokens: int) -> List[Chunk]:
    """将过大的 chunk 按段落边界拆分"""
    result = []
    for chunk in chunks:
        if chunk.token_estimate <= max_tokens:
            result.append(chunk)
            continue

        paragraphs = re.split(r'\n\n+', chunk.content)
        current_text = ""
        part_idx = 0

        for para in paragraphs:
            candidate = (current_text + '\n\n' + para).strip() if current_text else para
            if _estimate_tokens(candidate) > max_tokens and current_text:
                suffix = f"(续{part_idx})" if part_idx > 0 else ""
                result.append(Chunk(
                    title=chunk.title + suffix,
                    content=current_text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    heading_level=chunk.heading_level,
                    token_estimate=_estimate_tokens(current_text),
                ))
                part_idx += 1
                current_text = para
            else:
                current_text = candidate

        if current_text.strip():
            suffix = f"(续{part_idx})" if part_idx > 0 else ""
            result.append(Chunk(
                title=chunk.title + suffix,
                content=current_text,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                heading_level=chunk.heading_level,
                token_estimate=_estimate_tokens(current_text),
            ))
    return result


# ============================================================
#  Step 6: 表格记录 → Chunk
# ============================================================

def _record_to_text(record: dict, columns: list) -> str:
    """将一条表格记录转为 key: value 文本, 跳过空值字段"""
    parts: List[str] = []
    handled: set = set()

    for key in columns:
        if key not in record:
            continue
        handled.add(key)
        value = record[key]
        if isinstance(value, list):
            value = " | ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = "; ".join(f"{k}: {v}" for k, v in value.items())
        value = str(value).strip()
        if not value:
            continue
        parts.append(f"{key}: {value}")

    for key, value in record.items():
        if key in handled:
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            for si, sub in enumerate(value):
                sub_text = "; ".join(
                    f"{k}: {v}" for k, v in sub.items() if str(v).strip()
                )
                if sub_text:
                    parts.append(f"  {key}{si+1}: {sub_text}")
        elif isinstance(value, list):
            parts.append(f"{key}: {' | '.join(str(v) for v in value)}")
        else:
            value = str(value).strip()
            if value:
                parts.append(f"{key}: {value}")

    return "\n".join(parts)


def _build_table_record_chunks(
    table_info: dict,
    max_tokens: int = CHUNK_MAX_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
) -> List[Chunk]:
    """将一个表格的所有记录转为 Chunk 列表, 应用与文本 chunk 相同的大小约束"""
    records = table_info["records"]
    columns = table_info["columns"]
    table_name = table_info["table_name"]
    table_line = table_info.get("start_line", 0)

    if not records:
        return []

    raw_chunks: List[Chunk] = []
    for idx, record in enumerate(records):
        content = _record_to_text(record, columns)
        first_vals = [str(v) for v in record.values() if str(v).strip()][:2]
        title_hint = " - ".join(first_vals) if first_vals else f"#{idx+1}"

        raw_chunks.append(Chunk(
            title=f"{table_name} | {title_hint}",
            content=content,
            start_line=table_line,
            end_line=table_line,
            heading_level=99,
            token_estimate=_estimate_tokens(content),
        ))

    chunks = _merge_small_chunks(raw_chunks, min_tokens)
    chunks = _split_large_chunks(chunks, max_tokens)
    chunks = _merge_small_chunks(chunks, min_tokens)
    return chunks


def _split_oversized_table_blocks(
    table_blocks: List[Block],
    max_tokens: int = BLOCK_MAX_TOKENS,
    min_tokens: int = BLOCK_MIN_TOKENS,
) -> List[Block]:
    """将过大的表格 Block 拆分为多个子 Block, 过小的合并

    与文本 Block 不同, 表格 Block 内部没有子标题层级,
    因此按累积 token 数分组: 逐个 chunk 累加直到逼近 max_tokens 就切一刀。
    """
    # 先拆分
    expanded: List[Block] = []
    for block in table_blocks:
        total = sum(c.token_estimate for c in block.chunks)
        if total <= max_tokens:
            expanded.append(block)
            continue

        current_chunks: List[Chunk] = []
        current_tokens = 0
        part_idx = 0

        for chunk in block.chunks:
            if current_tokens + chunk.token_estimate > max_tokens and current_chunks:
                suffix = f"(第{part_idx + 1}部分)"
                expanded.append(Block(
                    title=block.title + suffix,
                    chunks=current_chunks,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    heading_level=block.heading_level,
                    source_file=block.source_file,
                    breadcrumb=list(block.breadcrumb),
                ))
                current_chunks = []
                current_tokens = 0
                part_idx += 1

            current_chunks.append(chunk)
            current_tokens += chunk.token_estimate

        if current_chunks:
            suffix = f"(第{part_idx + 1}部分)" if part_idx > 0 else ""
            expanded.append(Block(
                title=block.title + suffix,
                chunks=current_chunks,
                start_line=block.start_line,
                end_line=block.end_line,
                heading_level=block.heading_level,
                source_file=block.source_file,
                breadcrumb=list(block.breadcrumb),
            ))

    # 再合并过小的 (相邻且同表的 block)
    if len(expanded) <= 1:
        return expanded

    merged: List[Block] = []
    acc = expanded[0]
    acc_tokens = sum(c.token_estimate for c in acc.chunks)

    for blk in expanded[1:]:
        blk_tokens = sum(c.token_estimate for c in blk.chunks)
        same_table = (acc.start_line == blk.start_line
                      and acc.source_file == blk.source_file)
        if acc_tokens < min_tokens and same_table:
            acc.chunks.extend(blk.chunks)
            acc_tokens += blk_tokens
            acc.title = blk.title
        else:
            merged.append(acc)
            acc = blk
            acc_tokens = blk_tokens

    merged.append(acc)
    return merged


# ============================================================
#  主流程: 处理单个文件
# ============================================================

def process_file(
    filepath: str,
    block_level_threshold: int = BLOCK_LEVEL_THRESHOLD,
    chunk_max_tokens: int = CHUNK_MAX_TOKENS,
    chunk_min_tokens: int = CHUNK_MIN_TOKENS,
    block_max_tokens: int = BLOCK_MAX_TOKENS,
    block_min_tokens: int = BLOCK_MIN_TOKENS,
) -> List[Block]:
    """处理单个 .md 文件, 返回 Block 列表 (每个 Block 含若干 Chunk)"""
    path = Path(filepath)
    with open(path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    # Step 1-2: 逐行解析
    parsed = parse_document(raw_lines)

    # Step 3: 切 Block
    block_tuples = split_into_blocks(
        parsed, source_file=path.name,
        block_level_threshold=block_level_threshold,
    )

    # Step 3.5: 合并空壳 Block
    block_tuples = _merge_shell_blocks(block_tuples)

    # Step 3.6: 动态拆分超大 Block (按子标题层级迭代降级)
    block_tuples = _split_oversized_blocks(
        block_tuples, max_tokens=block_max_tokens)

    # Step 3.7: 合并过小 Block
    block_tuples = _merge_tiny_blocks(
        block_tuples, min_tokens=block_min_tokens)

    # Step 4-5: 每个 Block 内切 Chunk
    result: List[Block] = []
    for block, block_lines in block_tuples:
        block.chunks = split_block_into_chunks(
            block_lines,
            max_tokens=chunk_max_tokens,
            min_tokens=chunk_min_tokens,
        )
        if block.chunks:
            bc_prefix = " > ".join(block.breadcrumb)
            for chunk in block.chunks:
                chunk.content = bc_prefix + "\n" + chunk.content
                chunk.token_estimate = _estimate_tokens(chunk.content)
            result.append(block)

    # Step 6: 提取 HTML 表格, 每条记录生成独立 Chunk
    try:
        from process_table import parse_tables_from_text
    except ImportError:
        return result

    text = path.read_text(encoding='utf-8')
    table_infos = parse_tables_from_text(text)
    if not table_infos:
        return result

    # 字符偏移 → 行号 映射
    line_offsets: List[int] = []
    offset = 0
    for raw_line in raw_lines:
        line_offsets.append(offset)
        offset += len(raw_line)

    def _char_to_line(pos: int) -> int:
        lo, hi = 0, len(line_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_offsets[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1  # 1-based

    # 为每个表格确定面包屑上下文并构建 Block
    # 仅在 text block 中查找面包屑上下文, 避免表格 block 之间互相污染
    table_blocks: List[Block] = []
    for tinfo in table_infos:
        table_line = _char_to_line(tinfo["start_char"])
        tinfo["start_line"] = table_line
        table_name = tinfo["table_name"]

        chunks = _build_table_record_chunks(
            tinfo,
            max_tokens=chunk_max_tokens,
            min_tokens=chunk_min_tokens,
        )
        if not chunks:
            continue

        breadcrumb = [table_name]
        for blk in reversed(result):
            if blk.start_line <= table_line:
                breadcrumb = list(blk.breadcrumb) + [table_name]
                break

        bc_prefix = " > ".join(breadcrumb)
        for chunk in chunks:
            chunk.content = bc_prefix + "\n" + chunk.content
            chunk.token_estimate = _estimate_tokens(chunk.content)

        table_blocks.append(Block(
            title=table_name,
            chunks=chunks,
            start_line=table_line,
            end_line=table_line,
            heading_level=0,
            source_file=path.name,
            breadcrumb=breadcrumb,
        ))

    # Step 6.5: 表格 Block 大小约束 — 拆分过大、合并过小
    table_blocks = _split_oversized_table_blocks(
        table_blocks,
        max_tokens=block_max_tokens,
        min_tokens=block_min_tokens,
    )

    result.extend(table_blocks)
    result.sort(key=lambda b: b.start_line)

    return result


# ============================================================
#  输出: 终端展示 + JSON 导出
# ============================================================

def format_display(blocks: List[Block]) -> str:
    """生成人类可读的 Block-Chunk 结构概览"""
    lines = []
    total_chunks = 0

    for b_idx, block in enumerate(blocks):
        bc = " > ".join(block.breadcrumb)
        lines.append(f"{'=' * 70}")
        lines.append(
            f"BLOCK [{b_idx:02d}]  L{block.heading_level}  "
            f"line {block.start_line}-{block.end_line}  "
            f"({len(block.chunks)} chunks)"
        )
        lines.append(f"  标题: {block.title[:80]}")
        lines.append(f"  路径: {bc[:100]}")
        lines.append(f"  来源: {block.source_file}")

        for c_idx, chunk in enumerate(block.chunks):
            total_chunks += 1
            preview = chunk.content[:80].replace('\n', ' | ')
            is_last = c_idx == len(block.chunks) - 1
            connector = "└─" if is_last else "├─"
            lines.append(
                f"  {connector} CHUNK [{c_idx:02d}]  "
                f"~{chunk.token_estimate:>4d} tok  "
                f"line {chunk.start_line}-{chunk.end_line}"
            )
            lines.append(f"  {'  ' if is_last else '│ '}   {preview}...")
        lines.append("")

    lines.append(f"合计: {len(blocks)} Blocks, {total_chunks} Chunks")
    return '\n'.join(lines)


def blocks_to_dicts(blocks: List[Block]) -> List[dict]:
    """将 Block 列表转为可序列化的 dict 列表 (用于 JSON 导出)"""
    return [
        {
            'title': b.title,
            'source_file': b.source_file,
            'breadcrumb': b.breadcrumb,
            'heading_level': b.heading_level,
            'start_line': b.start_line,
            'end_line': b.end_line,
            'chunks': [
                {
                    'title': c.title,
                    'content': c.content,
                    'start_line': c.start_line,
                    'end_line': c.end_line,
                    'heading_level': c.heading_level,
                    'token_estimate': c.token_estimate,
                }
                for c in b.chunks
            ],
        }
        for b in blocks
    ]


# ============================================================
#  入口
# ============================================================

if __name__ == '__main__':
    import io
    # Windows 控制台默认 GBK, 强制 stdout/stderr 使用 UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob('*.md'))

    if not files:
        print("用法: python split_chunk.py [file1.md file2.md ...]")
        print("  或将 .md 文件放在当前目录下直接运行")
        sys.exit(1)

    all_blocks = []
    for fpath in files:
        print(f"\n{'─' * 70}")
        print(f"处理文件: {fpath}")
        print(f"{'─' * 70}")
        blocks = process_file(fpath)
        all_blocks.extend(blocks)
        print(format_display(blocks))

    # 导出 JSON
    output_path = 'chunks_output.json'
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(blocks_to_dicts(all_blocks), fp, ensure_ascii=False, indent=2)

    total_chunks = sum(len(b.chunks) for b in all_blocks)
    print(f"\n{'=' * 70}")
    print(f"全部完成: {len(all_blocks)} Blocks, {total_chunks} Chunks")
    print(f"JSON 已写入: {output_path}")
