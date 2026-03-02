# XR 生物安全语音助手 — 意图路由与 RAG 架构方案

## 一、场景概述

本系统为生物安全实验 XR 环境下的语音助手。用户在虚拟实验场景中通过语音提问，系统需判断用户意图并给出响应。

**完整链路**：

```text
用户语音 → ASR → Query 文本 → 意图路由 → [FuncCall / Prefill 直答 / RAG 增强回答] → TTS 播报
```

**核心约束**：

| 约束 | 说明 |
| :--- | :--- |
| LLM 部署 | SGLang，支持 RadixAttention 前缀缓存 |
| 模型 | Qwen3-30B-A3B（MoE 稀疏架构，推理时激活 ~3B，能力一般） |
| 实验知识 | 每个场景约 2000 汉字（~3000 token），可常驻 system prompt |
| 场景切换 | 前端主动发请求通知，后端据此更新前缀并预热 KV Cache |
| 等待体验 | 通过前端播放过渡语音（"正在思考中"）掩盖后端延迟 |

---

## 二、为什么不做扁平分类

最直觉的做法是把所有意图（4 个 funcall + "能否直答" + 8 个 RAG Role）拉平成一个十几类的分类问题，让关键词规则或 LLM 一次性判断。

**不这样做的理由**：

1. **funcall 和知识问答是完全不同性质的任务**。funcall 是对 XR 界面的操控指令，模式极其固定（"显示"、"切换"、"查看步骤"），用确定性规则匹配最可靠、延迟最低、不依赖 LLM。把它和语义理解任务混在一起会互相干扰。

2. **"Prefill 能否回答"不适合用规则判断**。判断已有上下文够不够回答一个问题，本质是语义理解任务。关键词规则只能粗糙地匹配实体，无法判断语义覆盖度。而 LLM 在前缀缓存场景下"看着答案判断能不能答"，天然比外层任何规则都准。

3. **Query 侧的 Role 分类可以搭 LLM 的便车**。既然知识问答路径必然要调用 LLM，让它在判断"答不了"的同时顺手输出所需的 Role 分类，比在外层单独维护一套 Query Role 规则系统更简洁。

因此采用**两层漏斗**架构，每层用最适合的技术。

---

## 三、整体架构

```text
                         User Query (ASR 输出)
                              │
                              ▼
                 ┌────────────────────────┐
                 │  第一层：关键词规则层    │
                 │  职责：FuncCall 识别    │
                 └────────────┬───────────┘
                              │
               ┌──── 命中 ────┤──── 未命中 ────┐
               ▼                               ▼
     返回 XR 指令 JSON                ┌────────────────────────┐
     (不调 LLM，延迟 ~0)             │  第二层：LLM 判断层     │
                                     │  前缀已缓存(实验知识)   │
                                     └────────────┬───────────┘
                                                  │
                                     ┌── 能答 ────┤──── 答不了 ────┐
                                     ▼                             ▼
                              LLM 直接回答                  LLM 输出需要的 Role
                              (快速路径)                    + 改写后的检索 Query
                                     │                             │
                                     ▼                             ▼
                                   TTS                     [RAG 检索管线]
                                                      按 Role 定向召回 Chunk
                                                                   │
                                                                   ▼
                                                           拼接到 prompt 末尾
                                                                   │
                                                                   ▼
                                                         [LLM 第二轮调用]
                                                        (前缀缓存仍命中)
                                                                   │
                                                                   ▼
                                                                 TTS
```

### 延迟分析

| 路径 | 预计延迟 | 占比预估 |
| :--- | :--- | :--- |
| **FuncCall 快速路径** | ~0ms（纯规则匹配） | ~15% 的请求 |
| **Prefill 直答路径** | LLM 生成延迟（前缀已缓存，首 token 快） | ~60% 的请求 |
| **RAG 路径** | LLM 首轮(短输出) + RAG 检索 + LLM 二轮生成 | ~25% 的请求 |

大部分请求（~75%）不需要 RAG 检索，延迟最优。

---

## 四、第一层：FuncCall 关键词规则

### 设计依据

funcall 意图有两个特征使其适合纯规则处理：
1. **触发模式极其固定**——用户在 XR 中请求界面操作时，用词集中在少数动作词 + 对象词的组合
2. **误判代价不对称**——漏判（该 funcall 的没识别出来）只是走了知识问答路径，用户会收到一个语音回答而非界面操作，体验下降但不严重；而把知识问答误判为 funcall 会导致错误的界面操作，体验更差

因此规则设计偏**高精度、可接受低召回**，宁可漏掉一些模糊表达让 LLM 兜底。

### 规则逻辑

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class FuncCallResult:
    command: str          # XR 指令名
    confidence: float     # 0~1
    params: dict          # 附加参数（如场景名）

# 每个 funcall 的触发模式：(正则模式, 排除模式, 指令名)
FUNCALL_RULES = [
    {
        "command": "ShowProcedurePanel",
        "triggers": [
            r"(查看|显示|打开).*(步骤|流程|进度)",
            r"(做到哪|到哪一步|第几步)(了|啦|呢)?$",
            r"(当前|现在).*(步骤|进度)",
        ],
        # 排除：带有条件情境、具体技术内容的不是 funcall
        "excludes": [
            r"(如果|万一|假如|遇到|发生)",
            r"(怎么做|怎么处理|怎么操作|怎么配|如何).{4,}",
        ],
    },
    {
        "command": "CurrentExperimentOperation",
        "triggers": [
            r"(当前|这一?步|现在).*(怎么[做操]|该[做干]什么)",
            r"(这步|这一步).*(是什么|做什么)",
        ],
        "excludes": [
            r"(如果|万一|假如|遇到|发生)",
            r".{6,}(怎么做|下一步)",   # 长句 + "怎么做" → 知识问答
        ],
    },
    {
        "command": "ShowEquipmentName",
        "triggers": [
            r"(这个?|那个?)(设备|仪器|器材|东西).*(叫什么|是什么|名[字称])",
            r"(显示|查看).*(仪器|设备).*(名[字称]|标签)",
        ],
        "excludes": [],
    },
    {
        "command": "SwitchExperimentScene",
        "triggers": [
            r"(切换|换|跳转|转到|进入).*(实验|场景|项目)",
            r"(我要?做|开始做?|打开).*实验",
        ],
        "excludes": [],
    },
]


def detect_funcall(query: str) -> Optional[FuncCallResult]:
    """检测 Query 是否为 FuncCall 意图

    遍历所有规则，命中 trigger 且不命中 exclude 则返回对应指令。
    多条规则命中时取第一个（规则按优先级排列）。
    """
    query_clean = query.strip()

    for rule in FUNCALL_RULES:
        # 先检查排除模式
        excluded = any(
            re.search(p, query_clean) for p in rule["excludes"]
        )
        if excluded:
            continue

        # 检查触发模式
        triggered = any(
            re.search(p, query_clean) for p in rule["triggers"]
        )
        if triggered:
            return FuncCallResult(
                command=rule["command"],
                confidence=0.95,
                params={},
            )

    return None
```

### 歧义场景处理

| 用户说的 | 判定 | 原因 |
| :--- | :--- | :--- |
| "现在第几步了" | FuncCall → `ShowProcedurePanel` | 短句 + 指示词"现在" + 无条件从句 |
| "这一步怎么操作" | FuncCall → `CurrentExperimentOperation` | "这一步" + 短句 |
| "加完样之后下一步怎么做" | **不命中 funcall** → 走 LLM | 包含具体技术内容（"加完样"），且超过长度排除阈值 |
| "如果离心机出了问题这步怎么办" | **不命中 funcall** → 走 LLM | 命中排除模式"如果" |
| "切换到 PCR 实验" | FuncCall → `SwitchExperimentScene` | 明确的切换动作 |

---

## 五、第二层：LLM 判断 + Role 路由

### 设计依据

在做法 A（外层规则预判 Prefill 能否回答）和做法 B（LLM 自行判断）之间选择了 **做法 B**，理由如下：

1. **SGLang 前缀缓存使 LLM 首轮调用成本极低**。实验知识（~3000 token）的 KV Cache 已经预热，首 token 延迟与空 prompt 接近。"多调一次 LLM"的代价比想象中小得多。

2. **LLM 判断"我能不能答"远比规则准确**。规则只能做实体匹配（Query 中的关键词是否出现在 Prefill 中），无法判断语义覆盖度。例如用户问"为什么要用碘液复染"，Prefill 中可能有革兰氏染色的完整步骤但没有解释原理——这种"有相关内容但答不了"的情况只有 LLM 能判断。

3. **Role 分类直接搭便车**。LLM 在判断"答不了"时顺手输出所需的 Role，省掉 Query 侧的关键词 Role 分类模块。这意味着 Query 侧完全不需要维护 Role 词表和打分规则，关键词打标只用于 Chunk 入库（离线、可容忍少量误差）。

4. **对模型能力要求可控**。虽然 Qwen3-30B-A3B 能力一般，但它只需要做两件事之一：要么直接回答问题（正常问答能力），要么输出一个简短的结构化标记（从 8 个 Role 中选 1~2 个）。不需要复杂推理。

### LLM 无法判断 Role 时的兜底

Qwen3-30B-A3B 能力有限，可能出现以下情况：
- 输出的 Role 不在 8 个合法值中
- 输出格式不符合约定
- 该说"答不了"的时候强行编造答案

**兜底策略**：对 LLM 返回的 `[NEED_RAG]` 标记做格式校验，如果 Role 解析失败，则回退到关键词规则对原始 Query 做 Role 打分（复用 Chunk 打标的 `score_role` 逻辑）。如果 LLM 没有输出 `[NEED_RAG]` 标记但回答质量可疑（如回答过短、包含"我不确定"等hedging表达），也触发 RAG 补充召回。

```text
LLM 第一轮输出
      │
      ├─ 正常回答（无 [NEED_RAG] 标记）
      │     │
      │     ├─ 回答质量 OK ──→ 直接返回 ✓
      │     │
      │     └─ 回答含 hedging 信号 ──→ 触发 RAG 补充（用关键词规则对 Query 做 Role 分类）
      │
      └─ 包含 [NEED_RAG:...] 标记
            │
            ├─ Role 解析成功 ──→ 按指定 Role 召回 ✓
            │
            └─ Role 解析失败 ──→ 回退关键词规则对 Query 打分取 Top-2 Role
```

### System Prompt 设计

```text
你是一名生物安全实验室的教学助手，正在指导学生完成虚拟实验。

## 当前实验场景知识
{experiment_knowledge}

## 回答规则
1. 如果上面的"当前实验场景知识"足够回答用户问题，直接给出清晰简洁的回答。
2. 如果上面的知识不够回答，或用户问的是当前实验以外的内容，输出以下格式（不要输出其他内容）：
   [NEED_RAG:role1,role2]
   QUERY:改写后的检索用 query

## Role 类别（选择最相关的 1~2 个）
- sop: 操作流程与步骤（怎么做、先后顺序）
- emergency: 应急处置（泄漏、暴露、事故处理）
- regulation: 法规制度（是否允许、合规要求）
- directory: 名录清单（病原体分类、分级）
- knowledge: 原理知识（概念解释、为什么）
- equipment: 设备说明（校准、维护、故障）
- reagent: 试剂信息（MSDS、保存、危害）
- notice: 通知公告（培训、考核、时间安排）

## 注意
- 不要编造你不确定的信息
- 如果犹豫是否能回答，倾向于输出 [NEED_RAG]
```

### 输出解析

```python
import re
import json
from typing import Optional, List
from dataclasses import dataclass

VALID_ROLES = {"sop", "emergency", "regulation", "directory",
               "knowledge", "equipment", "reagent", "notice"}

# LLM 回答中可能暗示不确定的信号词
HEDGING_SIGNALS = [
    "我不确定", "我不太清楚", "可能需要", "建议查阅",
    "具体请参考", "不在我的知识范围", "无法确定",
]

@dataclass
class LLMFirstPassResult:
    needs_rag: bool
    answer: Optional[str]          # 直答内容（needs_rag=False 时）
    roles: List[str]               # RAG Role 列表（needs_rag=True 时）
    rewritten_query: Optional[str] # LLM 改写后的检索 query
    hedging_detected: bool         # 是否检测到不确定信号


def parse_llm_output(output: str, original_query: str) -> LLMFirstPassResult:
    """解析 LLM 第一轮输出，判断走直答还是 RAG"""

    # 检查 [NEED_RAG:...] 标记
    rag_match = re.search(r'\[NEED_RAG:([^\]]+)\]', output)

    if rag_match:
        raw_roles = [r.strip().lower() for r in rag_match.group(1).split(",")]
        valid = [r for r in raw_roles if r in VALID_ROLES]

        # 提取改写后的 query
        query_match = re.search(r'QUERY[:：]\s*(.+)', output)
        rewritten = query_match.group(1).strip() if query_match else None

        if valid:
            return LLMFirstPassResult(
                needs_rag=True,
                answer=None,
                roles=valid,
                rewritten_query=rewritten or original_query,
                hedging_detected=False,
            )
        else:
            # Role 解析失败 → 回退关键词规则
            fallback_roles = keyword_role_classify(original_query, top_k=2)
            return LLMFirstPassResult(
                needs_rag=True,
                answer=None,
                roles=fallback_roles,
                rewritten_query=rewritten or original_query,
                hedging_detected=False,
            )

    # 没有 NEED_RAG 标记 → 检查是否存在 hedging 信号
    hedging = any(signal in output for signal in HEDGING_SIGNALS)

    return LLMFirstPassResult(
        needs_rag=False,
        answer=output,
        roles=[],
        rewritten_query=None,
        hedging_detected=hedging,
    )


def keyword_role_classify(query: str, top_k: int = 2) -> List[str]:
    """关键词规则对 Query 做 Role 分类（兜底用）
    复用 Chunk 入库时的 score_role 逻辑
    """
    scores = score_role(query, title="")  # score_role 来自已有的打标模块
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [role for role, _ in ranked[:top_k]]
```

---

## 六、SGLang 前缀缓存管理

### 场景切换时的预热流程

```python
class PrefixCacheManager:
    """管理 SGLang 的前缀缓存预热"""

    def __init__(self, sglang_url: str, system_prompt_template: str):
        self.sglang_url = sglang_url
        self.template = system_prompt_template
        self.current_scene: Optional[str] = None
        self.current_prefix: Optional[str] = None

    def on_scene_change(self, scene_id: str, experiment_knowledge: str):
        """前端通知场景切换时调用

        构建新的 system prompt 并发送预热请求，
        SGLang 会计算并缓存该前缀的 KV Cache。
        后续相同前缀的请求会自动命中缓存。
        """
        self.current_scene = scene_id
        self.current_prefix = self.template.format(
            experiment_knowledge=experiment_knowledge
        )

        # 发送预热请求（只计算 prefill，不生成）
        warmup_request = {
            "model": "qwen3-30b-a3b",
            "messages": [
                {"role": "system", "content": self.current_prefix}
            ],
            "max_tokens": 1,  # 最小生成量，目的只是触发前缀缓存
        }
        requests.post(f"{self.sglang_url}/v1/chat/completions",
                       json=warmup_request)

    def build_messages(self, query: str,
                       rag_context: Optional[str] = None) -> list:
        """构建完整的 messages 列表

        system prompt 部分保持不变以命中前缀缓存。
        RAG 上下文（如有）作为独立的 user message 拼入。
        """
        messages = [
            {"role": "system", "content": self.current_prefix},
        ]

        if rag_context:
            messages.append({
                "role": "user",
                "content": f"以下是相关参考资料：\n{rag_context}"
                           f"\n\n请根据以上资料回答用户问题。"
            })

        messages.append({"role": "user", "content": query})
        return messages
```

### 关键细节：为什么 RAG 上下文不放进 system prompt

RAG 召回的内容每次都不同，如果拼到 system prompt 中会**破坏前缀缓存命中**。因此：
- **system prompt**（固定不变）= 指令 + 实验知识 → **命中缓存**
- **RAG 上下文**（每次不同）= 作为 user message 追加 → **不影响缓存前缀**

这样即使走 RAG 路径的第二轮调用，system prompt 部分的 KV Cache 仍然复用。

---

## 七、文档分块算法

本系统采用专为结构化长文本设计的 **"文档 → Block（语义块） → Chunk（检索块）"** 自顶向下两级分块算法（实现见 `split_chunk.py` / `process_table.py`），解决传统文本切分工具在处理复杂层级 Markdown 文档时容易破坏语义完整性、丢失上下文的问题。

### 核心处理流程

```text
// 1. 配置常量
BLOCK_LEVEL_THRESHOLD = 2  // Block 的标题层级阈值 (例如 1 级和 2 级标题作为 Block 边界)
CHUNK_MAX_TOKENS = 384     // Chunk 最大 Token 数限制
CHUNK_MIN_TOKENS = 128     // Chunk 最小 Token 数限制
BLOCK_MAX_TOKENS = 2000    // Block 最大 Token 数限制
BLOCK_MIN_TOKENS = 100     // Block 最小 Token 数限制

函数 ProcessDocument(markdown_lines):
    // Step 1 & 2: 预处理与行级解析
    parsed_lines = 解析并清洗文档(markdown_lines)
    // 动作包含：清理 OCR/LaTeX 残留、过滤页眉/脚等噪声行、识别每一行的标题层级(支持 MD 语法、中文序号、多级数字编号)

    // Step 3: Block 划分 (基于高层级标题切分大语义主题)
    blocks = 按照阈值 (level <= BLOCK_LEVEL_THRESHOLD) 将 parsed_lines 切分为初始 Blocks
    对于 block 在 blocks 中:
        维护并生成当前 block 的 breadcrumb (面包屑路径，例如 "第一章 > 1.1")

    // Step 3 优化: Block 自适应合并与拆分
    // 3.1 合并空壳父级标题 (例如 "# 六、应急处理" 下紧跟 "## （一）总则"，将前者作为路径合并到后者)
    blocks = MergeShellBlocks(blocks)
    
    // 3.2 动态拆分超大 Block (向下降级拆分)
    当 存在总 Token > BLOCK_MAX_TOKENS 的 block 时:
        找到 block 内最浅的子标题层级
        以此层级为边界，将超大 block 拆分为多个子 blocks，并更新面包屑
        
    // 3.3 合并过小 Block (防止碎片化)
    blocks = MergeTinyBlocks(blocks) // 同一顶级章节内，合并过小的片段

    // Step 4 & 5: Chunk 划分 (Block 内部切分最小检索单元)
    final_chunks = []
    对于 block 在 blocks 中:
        // 4.1 寻找 Chunk 边界
        chunk_boundaries = []
        对于 line 在 block.lines 中:
            如果 line 是子标题 或 是"实质性长内容"的编号条目 (例如内容长度 >= 50 字符):
                chunk_boundaries.添加(line)
                
        // 4.2 初始切分
        raw_chunks = 根据 chunk_boundaries 切分 block 得到初始 chunks
        
        // 4.3 大小约束与平滑 (合并-拆分-合并)
        chunks = MergeSmallChunks(raw_chunks, CHUNK_MIN_TOKENS) // 将过小 chunk 与相邻 chunk 合并
        chunks = SplitLargeChunks(chunks, CHUNK_MAX_TOKENS)     // 将过大且无子标题的 chunk 按段落强行拆分
        chunks = MergeSmallChunks(chunks, CHUNK_MIN_TOKENS)     // 再次清理强行拆分可能产生的碎片
        
        // 4.4 面包屑注入 (为 RAG 检索赋能)
        对于 chunk 在 chunks 中:
            chunk.content = 拼合(block.breadcrumb) + "\n" + chunk.content
            final_chunks.添加(chunk)

    // Step 6: 提取与处理 HTML 表格
    table_blocks = []
    对于 table 在 解析出的所有表格 中:
        records = 提取表格记录(table, 支持跨行/跨列解析与前置列聚合)
        chunks = 将每条 record 转为独立的 chunk 文本 ("字段名: 字段值")
        chunks = 应用与文本 Chunk 相同的大小约束 (过小合并、过大拆分)
        table_block = 构建表格 Block(继承文档中该位置的面包屑路径, 包含上述 chunks)
        如果 table_block 过大:
            按 token 阈值将其拆分为多个子 Block (添加 "第N部分" 后缀)
        table_blocks.添加(table_block)

    将 table_blocks 按在文档中的物理行号插入到 blocks 中
    返回 final_chunks (包含文本 chunks 和表格 chunks)
```

### 与普通递归分块（Recursive Chunking）的区别与优势

在 RAG 应用中，常用的普通递归分块（例如 LangChain 的 `RecursiveCharacterTextSplitter`）通常基于预设的字符分隔符（如 `["\n\n", "\n", " ", ""]`）和固定的 Token 长度进行纯文本切片。与之相比，本算法在结构化文档处理上具有显著优势：

| 特性对比 | 普通递归分块 (Recursive Chunking) | 本算法 (Semantic Block-Chunking) |
| :--- | :--- | :--- |
| **切分依据** | 纯粹基于长度和换行符，属于"物理切分" | 基于文档本身的逻辑结构（Markdown 标题、中文层级、数字编号），属于"语义切分" |
| **上下文感知能力** | 极弱。切断后丢失上下文，大模型不知道该段落属于哪一章哪一节 | 极强。维护全局树状状态，每个 Chunk 强制注入其在文档树中的**完整路径（面包屑）** |
| **碎片控制** | 容易切出只有半句话或一两行的无意义 Chunk | 具有完善的"过小合并"机制，杜绝逻辑碎片 |
| **层级嵌套处理** | 扁平化处理，不区分父标题和子内容，标题可能与正文分离 | 具有父标题（空壳）合并机制，保留父子从属关系，确保标题永远引领其正文 |

**核心优势详解**：

1. **彻底解决 RAG 检索中的"上下文丢失"痛点**
   普通算法搜到一句话"*操作时必须穿戴防护服*"，大模型无法判断这是适用于 BSL-1 还是 BSL-3 实验室。本算法通过在 Chunk 首部打上 `第三章 实验室要求 > 3.2 BSL-3实验室` 的前缀，完美保留了全局语境，大幅提升向量检索的召回准确率。
2. **保证具体条款的语义完整性**
   利用"短列表项不切分、长编号条目才切分"的策略。例如，遇到 `1. 试管; 2. 离心机` 这种短枚举，算法会判定它们属于同一个语义段而拒绝切分；只有遇到 `1. 当实验室发生严重泄漏时，应当采取以下措施...` 这种长文本条目，才会将其作为独立的检索单元。
3. **消除"悬空标题"**
   普通基于长度的算法经常把标题切在一个 Chunk 的末尾，而将真正的正文切在下一个 Chunk 的开头。本算法严格以标题为边界，确保标题和它所管辖的正文始终绑定在一起。

### 核心算法特色点

#### 1. 面包屑路径的追踪与注入 (Breadcrumb Injection)

- **原理**：在逐行解析文档时，算法内部维护了一个**标题层级栈**（Level Stack）。
- **应用**：在最终生成最底层的 Chunk 时，将这个栈的路径拼接为字符串，**作为前缀强制注入到该 Chunk 内容的首部**。
- **价值**：极大提升了向量数据库的召回效果，使得最底层的细节操作也能因为包含了外层路径信息而被精准命中。

#### 2. "空壳标题"的智能合并 (Shell Block Merging)

- **痛点场景**：如"# 五、应急处理程序"下紧跟"## （一）总则"，空壳标题会成为废块。
- **算法特色**：自动识别空壳标题，**将其合并到下一个子 Block 的面包屑路径中，并销毁自身实体**。

#### 3. "过大拆分与过小合并"的自适应弹性呼吸机制 (Adaptive Sizing)

- **Block 级动态降维拆分**：寻找超大块内部最浅层级的子标题作为边界进行降维分解。
- **Chunk 级"三步约束" (Merge → Split → Merge)**：先合并过小片段，再针对无子标题的超长段落进行间距硬拆分，最后清理拆分产生的碎片。

#### 4. 复杂表格的结构化"解构与重组" (Structured Table Parsing)

- 将 HTML 表格的 `rowspan/colspan` 规范化，智能聚合左侧表头与数据，重构为 `字段名: 字段值` 的 Record-level Chunk，并注入面包屑。

---

## 八、Chunk 入库 Role 打标

### 设计依据

Chunk 侧采用**单 Role 存储**（每个 Chunk 仅分配一个 Role），理由如下：

1. **存储与索引简单**。单值字段可直接作为向量数据库的 metadata filter，检索时按 `role = "emergency"` 过滤即可，无需处理多值匹配。
2. **绝大多数 Chunk 确实属于单一 Role**。经过语义分块后，每个 Chunk 的内容通常是同质的（一个操作步骤、一条法规、一个设备说明）。
3. **多 Role 需求在 Query 侧解决**。如果用户问题跨越多个 Role，由 Query 侧输出多个 Role 去各自的池中召回，最终融合。不需要 Chunk 侧承担多标签的复杂度。

### 打标流程

```text
Chunk 文本 + 面包屑路径
         │
         ▼
  [关键词规则打分]  score_role(text, heading_path)
         │
         ├─ 置信度高 (分差 ≥ 3) ──→ 直接取 Top-1 Role 入库
         │
         └─ 置信度低 (分差 < 3)  ──→ 调 LLM 判断（离线批量，不影响在线延迟）
                                          │
                                          ▼
                                     取 LLM 返回的 Role 入库
```

### 关键词规则打分公式

给定一段 Chunk 文本 $t$、其所在的面包屑标题路径 $h$，对每个候选角色 $r \in \mathcal{R}$ 计算得分：

$$
S(r;\, t,\, h) \;=\; S_{\text{kw}}(r,\, t) \;+\; S_{\text{head}}(r,\, h) \;+\; S_{\text{struct}}(r,\, t)
$$

其中
$$
\mathcal{R}=\{\texttt{sop},\,\texttt{emergency},\,\texttt{regulation},\,\texttt{directory},\,\texttt{knowledge},\,\texttt{equipment},\,\texttt{reagent},\,\texttt{notice}\}.
$$

---

## ① 内容关键词得分 $S_{\text{kw}}$

每个角色 $r$ 维护一份关键词表 $K_r=\{(k_i,w_i)\}$，其中 $k_i$ 为关键词，$w_i\in\{1,2,3,4,5\}$ 为权重。统计关键词在文本中的出现次数并截断到上限 $C_{\max}=3$，防止单一高频词垄断得分：

$$
S_{\text{kw}}(r,\, t)
=\sum_{(k_i,\, w_i)\in K_r} w_i \cdot \min\!\bigl(\mathrm{count}(k_i,\, t),\; C_{\max}\bigr).
$$

---

## ② 标题路径加权得分 $S_{\text{head}}$

面包屑路径（如 `操作步骤 > 琼脂糖核酸电泳`）是比正文更强的角色信号。每个角色维护标题关键词表 $H_r=\{(k_j,w_j)\}$，命中时权重乘以放大系数 $\alpha=2$：

$$
S_{\text{head}}(r,\, h)
= \alpha \sum_{(k_j,\, w_j)\in H_r} w_j \cdot \mathbf{1}\{\,k_j \in h\,\}.
$$

其中 $k_j \in h$ 表示关键词 $k_j$ 是标题路径字符串 $h$ 的子串，$\mathbf{1}\{\cdot\}$ 为指示函数（条件成立取 1，否则取 0）。

---

## ③ 结构特征得分 $S_{\text{struct}}$

利用文本排版格式特征（与领域无关）加分。定义以下辅助量：

- $n_{\text{step}}(t)$：文本 $t$ 中编号步骤模式（`1.` `（1）` `一、` `第一步` `Step 1` 等）的匹配次数  
- $\mathrm{hasDef}(t)$：文本 $t$ 中是否含有术语定义模式（`2.1 XXX` 式层级编号，或含“术语”“定义”“是指”等）  
- $n_{\text{tone}}(t)$：规范性语气词（“必须”“应当”“不得”“禁止”“严禁”“违反”）在文本 $t$ 中的总出现次数  

则各角色的结构加分为（未列出的角色得 0 分）：

$$
S_{\text{struct}}(\texttt{sop},\, t)=
\begin{cases}
4, & n_{\text{step}}(t)\ge 3,\\
2, & 1\le n_{\text{step}}(t)<3,\\
0, & \text{otherwise}.
\end{cases}
$$

$$
S_{\text{struct}}(\texttt{knowledge},\, t)=
\begin{cases}
3, & \mathrm{hasDef}(t)=\text{true},\\
0, & \text{otherwise}.
\end{cases}
$$

$$
S_{\text{struct}}(\texttt{regulation},\, t)=
\begin{cases}
3, & n_{\text{tone}}(t)\ge 3,\\
1, & 1\le n_{\text{tone}}(t)<3,\\
0, & \text{otherwise}.
\end{cases}
$$

---

## 角色判定与置信度

取得分最高的角色为候选，置信度定义为 Top-1 与 Top-2 得分之差：

$$
r^{*}=\arg\max_{r\in\mathcal{R}} S(r;\, t,\, h),
$$

$$
\mathrm{conf}(t,\, h)
= S(r^{*};\, t,\, h) - \max_{r\in\mathcal{R},\, r\ne r^{*}} S(r;\, t,\, h).
$$

当 $\mathrm{conf}(t, h)\ge \theta$（默认 $\theta=3$）时直接采信规则结果；否则交由 LLM 批量打标。

---

### 附表：各 Role 关键词权重明细

> **权重分级标准**：按关键词对角色的**排他性**（specificity）分为三档：
>
> | 权重 | 含义 | 分级依据 |
> | :--- | :--- | :--- |
> | **4–5** | 强标识词 | 几乎只出现在该角色的文档中，具有排他性。如"应急预案"几乎不可能出现在 SOP 或 knowledge 文档中。 |
> | **2–3** | 中等指示词 | 在该角色文档中高频出现，但也可能少量出现在其他角色中。如"处置"在 emergency 中常见，但 regulation 中也偶尔提及。 |
> | **1** | 弱辅助词 | 单独出现不说明问题，但与其他中/强信号共现时提供累积证据。如"然后""记录"。 |

#### 表 1：sop（操作流程）

| 关键词 | 权重 | 分类 |
| :--- | :---: | :--- |
| 标准操作规程 | 4 | 文档类型直接标识 |
| 操作规程、标准操作、SOP/sop、操作步骤、操作流程 | 3 | 文档类型直接标识 |
| 操作方法、实验步骤、实验流程、实验方法 | 2 | 文档类型直接标识 |
| 按照以下步骤 | 3 | 通用程序性结构词 |
| 先后顺序、完成后、第一步/第二步/第三步 | 2 | 通用程序性结构词 |
| 准备工作、操作要点、操作要求 | 2 | 通用程序性结构词 |
| 目的与范围、记录表、记录表格 | 2 | SOP 文档通用章节名 |
| 依次、然后、接着、随后、注意事项、适用范围、职责 | 1 | 弱辅助词 |
| 记录、检查、确认、清洁、消毒、处理 | 1 | 通用操作动词 |

#### 表 2：emergency（应急处置）

| 关键词 | 权重 | 分类 |
| :--- | :---: | :--- |
| 应急预案 | 5 | 强标识 |
| 应急、应急处置、生物安全事故 | 4 | 强标识 |
| 预案、应急响应、泄漏、溢洒、暴露、事故、紧急、职业暴露、意外暴露、急救、疏散 | 3 | 中等指示 |
| 处置、感染、伤害、警报、消毒处理 | 2 | 中等指示 |
| 报告 | 1 | 弱辅助 |

#### 表 3：regulation（法规制度）

| 关键词 | 权重 | 分类 |
| :--- | :---: | :--- |
| 生物安全法 | 4 | 强标识 |
| 法规、条例、管理办法、管理制度、管理规定、合规要求 | 3 | 法规类型标识 |
| 制度、规定、合规、禁止、不得、审批、许可、资质、备案、报批、监督管理、安全管理、实验室管理、病原微生物实验室 | 2 | 中等指示 |
| 应当、必须、要求 | 1 | 弱辅助（规范语气） |

#### 表 4：directory（名录清单）

| 关键词 | 权重 | 分类 |
| :--- | :---: | :--- |
| 名录 | 5 | 强标识 |
| 清单、目录、第一类/第二类/第三类/第四类、高致病性 | 3 | 分类分级体系 |
| 分类、分级、人间传染、BSL-1/2/3/4、病原微生物、病原体、菌种、毒种、毒株 | 2 | 中等指示 |

#### 表 5：knowledge（原理知识）

| 关键词 | 权重 | 分类 |
| :--- | :---: | :--- |
| 术语和定义 | 4 | 强标识 |
| 原理、概念、定义、术语 | 3 | 认知/解释性标识 |
| 知识、机制、机理、理论、是什么、为什么、是指、称为 | 2 | 说明/论述性标识 |
| 解释、因为、由于、即、结构、功能、特征、特性、分类、组成、作用 | 1 | 弱辅助 |

#### 表 6：equipment（设备仪器）

| 关键词 | 权重 | 分类 |
| :--- | :---: | :--- |
| 设备、仪器、校准、故障、设备管理、仪器管理 | 3 | 核心标识 |
| 维护、保养、维修、巡检、操作面板、使用说明、开机、关机、运行参数 | 2 | 设备管理动作 |

#### 表 7：reagent（试剂信息）

| 关键词 | 权重 | 分类 |
| :--- | :---: | :--- |
| MSDS、SDS | 4 | 强标识 |
| 试剂、安全数据表、安全技术说明书 | 3 | 核心标识 |
| 药品、化学品、试剂盒、试剂准备、危害、有毒、腐蚀、易燃、配方、有效期、保质期 | 2 | 中等指示 |
| 保存、存储、储存、浓度、稀释 | 1 | 弱辅助 |

#### 表 8：notice（通知公告）

| 关键词 | 权重 | 分类 |
| :--- | :---: | :--- |
| 通知、公告 | 4 | 强标识 |
| 培训、考核 | 3 | 行政活动标识 |
| 安排、会议、签到、报名 | 2 | 中等指示 |
| 时间、日期、年度、季度 | 1 | 弱辅助 |

---

### 权重设定的合理性论证

关键词打分的权重是人工设定的，这在答辩中必然面临"主观性"质疑。以下从四个角度论证其合理性。

#### 1. 权重反映的是"排他性"，有明确的语言学依据

权重并非任意赋值，而是遵循一条可操作化的分级准则——**排他性（specificity）**：

- **权重 4–5**（强标识）：该词在语料库中**几乎只出现在目标角色**的文档中。例如"应急预案"在非 emergency 文档中出现概率极低，反之"应急预案"出现则近乎必然是 emergency。这一判断可通过统计各角色语料中的文档频率（DF）来验证。
- **权重 2–3**（中等指示）：该词在目标角色中高频，但在其他角色中也会出现。例如"处置"在 emergency 中常见，regulation 中偶尔提及。
- **权重 1**（弱辅助）：该词的角色区分度低，但在多个同角色信号共现时提供累积证据。

这与信息检索领域 BM25 中 IDF（逆文档频率）的思想一致：越稀有的词对分类贡献越大。只是由于语料规模有限，采用人工分级代替统计计算。

#### 2. 系统对权重偏差具有鲁棒性

打分公式的设计使得结果对单个权重值的精确度不敏感：

- **置信度阈值 \(\theta\) 起到缓冲作用**。最终判定不依赖 Top-1 角色的绝对得分，而依赖 **Top-1 与 Top-2 的分差**。即使某个权重偏大或偏小，只要它不改变 Top-1 的排名，或分差仍然超过阈值，判定结果不变。
- **出现次数截断 \(\min(\text{count}, 3)\)** 防止单一高频词主导。即使一个权重为 1 的弱辅助词出现 100 次，其贡献也被限制在 3 分。
- **三路信号互相补偿**。内容关键词、标题路径、结构特征三路独立信号同时投票，某一路的权重偏差可被另外两路修正。

更关键的是：**低置信度的 Chunk 会自动转交 LLM 判断**。权重规则只需做到"有把握的判对、没把握的交出去"，不需要在边界情况上精确。阈值 \(\theta = 3\) 意味着至少要赢第二名 3 分才自信——这是一个保守的标准。

#### 3. 关键词选择遵循"标记角色格式、而非领域主题"的原则

一个常见的陷阱是在 SOP 关键词中塞入 "PCR""离心""电泳" 等领域特定实验术语。这些词说明的是**内容主题**（分子生物学）而非**文档角色**（操作规程），会导致：
- **假阳性**：一篇讲 PCR 原理的 knowledge 文章同样包含 "PCR""扩增"
- **假阴性**：废弃物处理 SOP、人员进出管理 SOP 不含这些词，会被漏判

本系统的关键词只选取标记**文档功能格式**的词汇（如"操作步骤""按照以下步骤""依次"），辅以不依赖领域的排版结构检测（编号步骤计数）。这一设计使规则在未见过的领域 SOP 上仍然有效。

#### 4. 可通过消融实验验证有效性

权重设定最终应以下游任务效果为评判标准。建议补充以下实验：

| 对比方案 | 做法 | 预期结果 |
| :--- | :--- | :--- |
| **本方案**（三档权重 + 结构特征） | 完整打分公式 | 基线 |
| **均匀权重**（所有关键词权重 = 1） | 去掉权重分级 | 低置信度 Chunk 比例上升，更多需 LLM 兜底 |
| **去除结构特征** | 仅用关键词 + 标题 | SOP/knowledge/regulation 的 P/R 下降 |
| **去除标题加权** | 仅用关键词 + 结构 | 同一文档内跨角色章节的区分度下降 |
| **纯 LLM** | 所有 Chunk 都调 LLM | 准确率可能最高，但离线成本大幅增加 |

在人工标注的测试集上对比各方案的 **Precision / Recall / F1**，以及**需 LLM 兜底的比例**（即系统效率），可以定量证明三档权重设计的边际收益。

---

### LLM 批量打标 Prompt

对于低置信度 Chunk，离线批量调用 LLM（可用更强的模型如 DeepSeek-V3，不受在线延迟约束）：

```text
请判断以下文本片段最适合归入哪个知识类别。只返回一个类别名称。

类别：sop / emergency / regulation / directory / knowledge / equipment / reagent / notice

文本片段的层级路径：{heading_path}
文本内容：
{chunk_text}

类别：
```

### 打标结果存储

```json
{
  "chunk_id": "doc1_block3_chunk2",
  "content": "第三章 实验室要求 > 3.2 BSL-3实验室\n操作时必须穿戴防护服...",
  "role": "regulation",
  "role_confidence": 5,
  "block_id": "doc1_block3",
  "block_role": "regulation",
  "embedding": [0.012, -0.034, ...],
  "metadata": {
    "source_file": "生物安全通用要求.md",
    "breadcrumb": ["第三章 实验室要求", "3.2 BSL-3实验室"],
    "start_line": 142,
    "end_line": 158
  }
}
```

---

## 九、完整请求处理流程

```python
class VoiceAssistant:
    """XR 语音助手主控制器"""

    def __init__(self, prefix_manager, retriever, sglang_client):
        self.prefix_mgr = prefix_manager
        self.retriever = retriever      # RAG 检索器
        self.llm = sglang_client

    def handle_query(self, query: str) -> dict:
        """处理一条用户 Query，返回响应"""

        # ========== 第一层：FuncCall 检测 ==========
        funcall = detect_funcall(query)
        if funcall:
            return {
                "type": "funcall",
                "command": funcall.command,
                "params": funcall.params,
            }

        # ========== 第二层：LLM 第一轮调用 ==========
        # 前缀已缓存，首 token 延迟低
        messages = self.prefix_mgr.build_messages(query)
        llm_output = self.llm.chat(messages, max_tokens=512)

        result = parse_llm_output(llm_output, query)

        # --- 直答路径 ---
        if not result.needs_rag and not result.hedging_detected:
            return {
                "type": "answer",
                "content": result.answer,
            }

        # --- Hedging 检测：LLM 没说 NEED_RAG 但回答不自信 ---
        if result.hedging_detected:
            roles = keyword_role_classify(query, top_k=2)
            search_query = query
        else:
            roles = result.roles
            search_query = result.rewritten_query or query

        # ========== RAG 检索 ==========
        chunks = self.retriever.search(
            query=search_query,
            roles=roles,
            top_k=8,
        )

        if not chunks:
            # 召回为空，直接返回 LLM 首轮回答（有总比没有好）
            if result.answer:
                return {"type": "answer", "content": result.answer}
            return {"type": "answer", "content": "抱歉，我暂时无法回答这个问题。"}

        # ========== LLM 第二轮调用（RAG 增强） ==========
        rag_context = "\n\n---\n\n".join(c["content"] for c in chunks)
        messages = self.prefix_mgr.build_messages(query, rag_context=rag_context)
        final_answer = self.llm.chat(messages, max_tokens=1024)

        return {
            "type": "answer",
            "content": final_answer,
            "sources": [c["chunk_id"] for c in chunks],
        }
```

---

## 十、模块分工总览

| 模块 | 技术手段 | 职责 | 在线/离线 |
| :--- | :--- | :--- | :--- |
| **FuncCall 识别** | 关键词正则规则 | 从 Query 中识别 4 种 XR 操作指令 | 在线，~0ms |
| **Prefill 直答判断** | LLM 自行判断 | 利用前缀中的实验知识直接回答 | 在线，LLM 首轮 |
| **Query Role 分类** | LLM 输出 `[NEED_RAG:roles]` | LLM 判断答不了时顺带输出 Role | 在线，LLM 首轮 |
| **Query Role 兜底** | 关键词规则 `score_role()` | LLM 输出格式异常时回退 | 在线，~0ms |
| **Chunk Role 打标** | 关键词规则 + LLM 批量兜底 | 入库时为每个 Chunk 分配单一 Role | 离线 |
| **前缀缓存管理** | SGLang RadixAttention | 实验知识预热、场景切换更新 | 场景切换时 |
| **RAG 检索** | 向量检索 + BM25，按 Role 过滤 | 定向召回相关 Chunk | 在线 |

### 关键词规则与 LLM 的分工原则

```text
确定性高、模式固定的 → 关键词规则（funcall 识别、Chunk 打标）
语义理解、需要判断力的 → LLM（能否直答、Query Role 分类）
LLM 不可靠时 → 关键词规则兜底（Role 解析失败、hedging 检测后的降级）
```

这样关键词规则只做它擅长的"模式匹配"，不试图理解语义；LLM 只做它擅长的"语义判断"，不浪费在确定性任务上。两者互为补充而非互相替代。

---

## 十一、后续可优化方向（暂不实施）

1. **模型升级**：Qwen3-30B-A3B 替换为能力更强的模型后，LLM 首轮的直答覆盖率和 Role 分类准确率都会提升，RAG 触发率进一步下降。
2. **Cross-Encoder 精排**：在 RAG 召回后增加重排序步骤，提升送入 LLM 的上下文质量。
3. **流式输出 + 提前截断**：LLM 第一轮如果前几个 token 就输出了 `[NEED_RAG`，可以立即截断生成并触发 RAG，不等完整输出。
4. **对话历史管理**：当前方案是单轮问答，后续可加入多轮对话的上下文管理（注意不要破坏前缀缓存的命中率）。
5. **数据飞轮**：记录每次 LLM 的 Role 分类结果，积累标注数据后可考虑训练轻量分类模型替代 LLM 首轮判断，进一步降低延迟。
6. **块级升格机制 (Block Promotion)**：向量/BM25 召回高相关 Chunk 后，检查其所属 Block 的 Role 是否一致；若一致则将整个 Block 升格召回，为 LLM 提供更完整的同质上下文。
