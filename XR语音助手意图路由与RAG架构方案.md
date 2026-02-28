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

## 七、Chunk 入库 Role 打标

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

## 八、完整请求处理流程

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

## 九、模块分工总览

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

## 十、后续可优化方向（暂不实施）

1. **模型升级**：Qwen3-30B-A3B 替换为能力更强的模型后，LLM 首轮的直答覆盖率和 Role 分类准确率都会提升，RAG 触发率进一步下降。
2. **Cross-Encoder 精排**：在 RAG 召回后增加重排序步骤，提升送入 LLM 的上下文质量。
3. **流式输出 + 提前截断**：LLM 第一轮如果前几个 token 就输出了 `[NEED_RAG`，可以立即截断生成并触发 RAG，不等完整输出。
4. **对话历史管理**：当前方案是单轮问答，后续可加入多轮对话的上下文管理（注意不要破坏前缀缓存的命中率）。
5. **数据飞轮**：记录每次 LLM 的 Role 分类结果，积累标注数据后可考虑训练轻量分类模型替代 LLM 首轮判断，进一步降低延迟。
