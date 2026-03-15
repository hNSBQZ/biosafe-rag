# Block 升格窗口管理策略讨论

## 一、现状参数一览

| 参数 | 值 | 来源 |
|:---|:---|:---|
| CHUNK_MAX_TOKENS | 384 | split_chunk.py |
| CHUNK_MIN_TOKENS | 128 | split_chunk.py |
| BLOCK_MAX_TOKENS | 2000 | split_chunk.py |
| BLOCK_MIN_TOKENS | 100 | split_chunk.py |
| 升格阈值 max_block_tokens | 2000 | retriever.py |
| top_k（最终返回 chunk 数） | 8 | query_handler.py |
| System prompt（实验场景知识） | ~3000 tokens | readme.md |
| LLM 第二轮 max_tokens | 1024 | readme.md |
| 模型 | Qwen3.5-27B | readme.md |

---

## 二、实际数据分布（来自 role_result.json）

### Chunk 分布（共 842 个）

| 指标 | 值 |
|:---|:---|
| 最小 | 8 tokens |
| 最大 | 2480 tokens |
| 中位数 | 233 tokens |
| 平均 | 256 tokens |

| 区间 | 数量 | 占比 |
|:---|:---|:---|
| [0, 100) | 37 | 4.4% |
| [100, 200) | 136 | 16.2% |
| **[200, 300)** | **575** | **68.3%** |
| [300, 400) | 42 | 5.0% |
| [400, 500) | 16 | 1.9% |
| [500, +∞) | 36 | 4.3% |

> 绝大多数 chunk 落在 200~300 tokens，分切质量良好。

### Block 分布（共 235 个）

| 指标 | 值 |
|:---|:---|
| 最小 | 8 tokens |
| 最大 | 2480 tokens |
| 中位数 | 595 tokens |
| 平均 | 917 tokens |
| 平均含 chunk 数 | 3.6 个 |

### 可升格 Block（共 111 个，占比 47%）

满足升格条件（≥2 chunks、单一 role、总 token ≤ 2000）的 block：

| 指标 | 值 |
|:---|:---|
| 最小 | 241 tokens |
| 最大 | 1997 tokens |
| **中位数** | **1794 tokens** |
| 平均 | 1351 tokens |

**role 分布：**

| role | 可升格 block 数 |
|:---|:---|
| directory | 67（60%）|
| regulation | 26（23%）|
| equipment | 6 |
| sop | 5 |
| emergency | 4 |
| reagent | 2 |
| knowledge | 1 |

> 关键发现：可升格 block 中位数高达 1794 tokens，接近 2000 上限。directory 类占大头（名录表格天然是大 block）。升格后一个 block 约等于 7 个普通 chunk 的体积。

---

## 三、Qwen3.5-27B 上下文窗口推荐

Qwen2.5/3 系列的原生训练上下文为 **32,768 tokens**，通过 YaRN 扩展可到 128K，但扩展区间的注意力质量下降。在 SGLang + RadixAttention 部署下：

| 窗口范围 | 注意力质量 | 推荐场景 |
|:---|:---|:---|
| **0 ~ 8K** | 最佳 | 核心推理区，system prompt + 用户问题 |
| **8K ~ 16K** | 良好 | RAG 上下文主区间，信息利用率高 |
| 16K ~ 32K | 可用但递减 | 长文档兜底，非关键补充材料 |
| 32K+ | 显著下降 | 不推荐用于需要精确推理的场景 |

**推荐有效工作窗口：≤ 16K tokens**（包含 system prompt + query + RAG 上下文 + 输出预留）。

---

## 四、Token 预算估算

### 预算分配表

以 **16K 有效窗口**为基准：

| 组成部分 | Token 占用 | 说明 |
|:---|:---|:---|
| System prompt（实验知识） | ~3000 | 固定，已由 SGLang 前缀缓存 |
| 用户 query | ~100 | 通常 20~50 字 |
| 格式开销（分隔符、指令） | ~200 | `---` 分隔、"参考资料："前缀等 |
| LLM 输出预留 | 1024 | max_tokens 设置 |
| **RAG 上下文预算** | **~11,900** | 16384 - 3000 - 100 - 200 - 1024 |

取整：**RAG 上下文预算 ≈ 12,000 tokens**。

### 场景估算

| 场景 | RAG tokens | 占预算比 | 是否安全 |
|:---|:---|:---|:---|
| **无升格**（8 × avg 256） | 2,048 | 17% | 非常安全 |
| **典型**（3 升格 + 5 普通 chunk） | ~7,253 | 61% | 安全 |
| **极端最坏**（8 个最大升格 block） | 15,837 | 133% | **超预算** |
| **较坏**（5 升格 + 3 普通 chunk） | ~10,638 | 89% | 勉强 |

> 极端最坏的场景（8 条结果全部升格为 ~2000 tokens 的 block）会突破 12K 预算。虽然实际中不太可能 8 个结果来自 8 个不同的大 block，但 5~6 个升格的情况是有可能发生的，此时已经接近上限。

---

## 五、三种候选策略对比

### 策略 A：减少 top_k

升格后减少最终返回数量。例如：升格了 N 个 block，就减少返回 N 个结果。

```
优点：实现最简单，一行代码
缺点：粗暴，可能丢失不同维度的相关结果
      升格本身是为了提供更完整的上下文，减少条目与之矛盾
```

**评价：不推荐。** 降低了召回多样性，RAG 的核心价值在于多角度覆盖。

### 策略 B：贪心填充（Token Budget）

设定一个 `max_context_tokens` 预算（如 10,000），从 RRF 分数最高的结果开始逐条填入，填满即止。

```python
budget = 10000
selected = []
used = 0
for result in sorted_results:
    tok = estimate_tokens(result["content"])
    if used + tok > budget:
        break
    selected.append(result)
    used += tok
```

```
优点：token 使用可控，绝不超窗口
      保留了 RRF 排序的优先级
缺点：尾部低分但可能有价值的结果被直接截断
      如果前几条都是大 block（~2000 tokens），可能只返回 4~5 条
```

**评价：简单有效，是一个好的基线方案。**

### 策略 C：贪心填充 + 降级（推荐）

在策略 B 基础上增加降级逻辑：当预算不足以放入一个完整的升格 block 时，不是直接跳过，而是**降级为只放入被召回的那个原始 chunk**。

```python
budget = 10000
selected = []
used = 0
for result in sorted_results:
    tok = estimate_tokens(result["content"])
    if used + tok <= budget:
        selected.append(result)
        used += tok
    elif result.get("promoted"):
        # 降级：升格 block 放不下，退回原始 chunk
        original_chunk = get_original_chunk(result)
        chunk_tok = estimate_tokens(original_chunk["content"])
        if used + chunk_tok <= budget:
            selected.append(original_chunk)
            used += chunk_tok
    # 非升格且放不下的直接跳过
```

```
优点：前面的高分结果优先享受升格的完整上下文
      后面装不下的自动降级为单 chunk，不丢信息
      token 使用严格可控
      最大化了「高分完整、低分至少有」的信息密度
缺点：实现稍复杂，需保留原始 chunk 信息用于降级
```

**评价：推荐。** 兼顾了窗口安全和信息完整性。

---

## 六、策略对比总结

| 维度 | A. 减 top_k | B. 贪心填充 | C. 贪心 + 降级 |
|:---|:---|:---|:---|
| 窗口安全 | 不可控 | 严格可控 | 严格可控 |
| 信息密度 | 低（丢条目） | 中（丢尾部） | **高（尾部降级保留）** |
| 召回多样性 | 差 | 中 | **好** |
| 实现复杂度 | 低 | 低 | 中 |
| 升格收益保留 | 全保留 | 全保留 | 高分全保留，低分降级 |

---

## 七、推荐方案：策略 C 实现要点

### 参数建议

| 参数 | 建议值 | 说明 |
|:---|:---|:---|
| `max_context_tokens` | **10,000** | 12K 预算留 2K 安全余量 |
| `max_block_tokens`（升格阈值） | **2000**（不变） | 当前值合理 |
| `top_k`（初始召回数） | **12** | 适当放大，给贪心填充更多候选 |

### 核心改动

1. **Retriever.search()** 增加 `max_context_tokens` 参数
2. 升格完成后，新增一步 **token 预算裁剪**：
   - 按 RRF score 从高到低遍历
   - 放得下 → 整条放入
   - 放不下且是升格 block → 降级为原始 chunk 再尝试
   - 放不下且是普通 chunk → 跳过
3. 升格时在 result 中保留 `original_content` 字段，供降级使用
4. 日志输出裁剪过程（哪些被完整保留、哪些被降级、哪些被丢弃）

### 为什么 top_k 建议放大到 12

当前 top_k=8 是"最终给 LLM 的数量"。引入预算裁剪后，实际给 LLM 的数量由预算决定，top_k 变成了"候选池大小"。放大到 12 可以：
- 让贪心填充有更多候选，提高信息覆盖
- 即使前面几条升格后吃了大量预算，后面仍有足够的小 chunk 可以填入

---

## 八、极端 Case 验证

以推荐方案（budget=10,000, top_k=12）模拟：

**Case 1：名录查询（高升格率）**

如"新冠病毒是哪类病原体"，directory role 下 block 大且密集。

```
[1] 升格 block 1997 tok → 放入 (累计 1997)
[2] 升格 block 1984 tok → 放入 (累计 3981)
[3] 升格 block 1976 tok → 放入 (累计 5957)
[4] 升格 block 1973 tok → 放入 (累计 7930)
[5] 升格 block 1969 tok → 放不下，降级为原始 chunk 256 tok (累计 8186)
[6] 升格 block 1968 tok → 放不下，降级为原始 chunk 256 tok (累计 8442)
[7] 普通 chunk 233 tok → 放入 (累计 8675)
[8] 普通 chunk 233 tok → 放入 (累计 8908)
```

结果：4 个完整 block + 2 个降级 chunk + 2 个普通 chunk = 8908 tokens，安全。

**Case 2：SOP 查询（低升格率）**

如"核酸提取的操作步骤"，sop role 下 block 较小，升格少。

```
[1] 升格 block 800 tok → 放入 (累计 800)
[2] 普通 chunk 256 tok → 放入 (累计 1056)
...
[8] 普通 chunk 256 tok → 放入 (累计 2856)
```

结果：1 个升格 block + 7 个普通 chunk = 2856 tokens，绰绰有余。

---

## 九、结论

1. **当前 12K 的 RAG 预算在典型场景下是安全的**，但极端升格场景存在超窗口风险
2. **推荐策略 C（贪心填充 + 降级）**，在 retriever 的 `_promote_blocks` 之后增加一步 token 预算裁剪
3. `max_context_tokens` 建议设为 **10,000**（留 2K 安全余量）
4. `top_k` 建议从 8 放大到 **12**，让候选池更充裕
5. 实现改动集中在 `retriever.py` 的 `search()` 方法尾部，对现有逻辑侵入小
