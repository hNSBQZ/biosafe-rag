# Intent Router 技术方案

## 一、问题定义

在当前 RAG 系统中，用户 Query 需要被路由到 8 个 Role 池（sop / emergency / regulation / directory / knowledge / equipment / reagent / notice）中的一个或多个，以实现定向检索降噪。这一环节即 **Intent Router（意图路由器）**。

Intent Router 的核心任务：

| 输入 | 输出 |
| :--- | :--- |
| 用户原始 Query（中文自然语言） | 1~N 个目标 Role + 各自置信度 |

### 关键约束

- **延迟**：Intent Router 处于检索链路最前端，P99 延迟需控制在 **50ms 以内**（本地推理）或 **200ms 以内**（含网络调用）。
- **准确率**：Top-1 Role 准确率目标 ≥ 90%，Top-2 覆盖率目标 ≥ 97%。
- **多意图支持**：单条 Query 可能同时命中多个 Role（如"BSL-3 实验室泄漏后怎么处理，有什么制度要求？"同时涉及 emergency + regulation）。
- **冷启动可行性**：初期可能缺少大量标注数据，方案需支持从零起步逐步优化。

---

## 二、可选技术路线对比

| 方案 | 原理 | 优势 | 劣势 | 适用阶段 |
| :--- | :--- | :--- | :--- | :--- |
| **A. 关键词规则** | 词表匹配 + 加权打分（现有方案） | 零依赖、零延迟、完全可控 | 脆弱、不泛化、维护成本高、无法处理同义改写 | 原型验证 |
| **B. LLM 零样本分类** | Prompt 让大模型直接输出 Role | 无需训练数据、理解力强 | 延迟高（500ms~2s）、成本高、输出不稳定 | 标注辅助 / 兜底 |
| **C. 轻量分类模型** | 在中文 BERT/BGE 上加分类头微调 | 精度高、推理快（<20ms）、支持多标签 | 需要标注数据（≥500 条） | 生产主力 |
| **D. 嵌入原型匹配** | 将 8 个 Role 表示为原型向量，Query 向量做余弦相似度 | 极少标注（每 Role 5~10 条即可）、部署简单 | 精度天花板低于微调模型 | 冷启动过渡 |
| **E. 混合级联** | D/A 做初筛 → C 做精排 → B 做兜底 | 兼顾冷启动和长期精度 | 工程复杂度较高 | 全生命周期 |

---

## 三、推荐方案：三阶段渐进式演进

### 总体架构

```text
User Query
    │
    ▼
┌─────────────────────┐
│  Stage 0: 预处理     │  术语补全 / 指代消解 / Query 改写
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 1: 快速初筛   │  嵌入原型匹配（冷启动）或 轻量分类模型（成熟期）
│  输出: Role 候选集   │  附带置信度分数
└─────────┬───────────┘
          │
          ▼  (置信度 < 阈值？)
┌─────────────────────┐
│  Stage 2: LLM 仲裁   │  仅对低置信度 Query 调用，输出精确 Role + 理由
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  多路召回 + 融合排序  │  按 Role 定向检索 → RRF 融合 → Cross-Encoder 精排
└─────────────────────┘
```

### 阶段一：冷启动期（0~500 条标注数据）

**技术选型：嵌入原型匹配（Prototype-based Classification）**

**核心思路**：为每个 Role 构造若干"原型 Query"，用 Embedding 模型编码后取平均得到 Role 原型向量。线上将用户 Query 编码后与 8 个原型做余弦相似度，取 Top-K。

```python
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-zh-v1.5"  # 中文嵌入模型，768 维
encoder = SentenceTransformer(MODEL_NAME)

ROLE_PROTOTYPES = {
    "sop": [
        "PCR 实验的操作步骤是什么",
        "怎么配制培养基",
        "离心操作的流程和注意事项",
        "革兰氏染色的先后步骤",
        "如何正确使用移液器",
    ],
    "emergency": [
        "实验室发生化学品泄漏怎么处理",
        "被针刺伤后应该怎么做",
        "发生病原微生物暴露事故如何上报",
        "实验室着火了怎么处置",
        "生物安全柜内发生泼溅怎么清理",
    ],
    "regulation": [
        "BSL-2 实验室需要满足什么管理要求",
        "废弃物处置有什么规定",
        "实验记录必须保留多长时间",
        "高致病性病原微生物运输需要什么审批",
        "实验室准入制度有哪些要求",
    ],
    "directory": [
        "新冠病毒属于第几类病原微生物",
        "布鲁氏菌的危害等级和实验室要求",
        "人间传染的病原微生物名录",
        "哪些病原体需要在 BSL-3 实验室操作",
        "炭疽芽孢杆菌的分类和英文名",
    ],
    "knowledge": [
        "什么是生物安全柜的工作原理",
        "高压灭菌的原理是什么",
        "PCR 扩增的基本概念",
        "为什么需要进行无菌操作",
        "生物安全等级的定义和区别",
    ],
    "equipment": [
        "离心机报警代码 E03 是什么意思",
        "超净工作台怎么校准",
        "生物安全柜的日常维护保养流程",
        "PCR 仪温度异常怎么排查",
        "高压灭菌器使用注意事项",
    ],
    "reagent": [
        "乙醇的 MSDS 安全信息",
        "福尔马林怎么保存，有效期多长",
        "次氯酸钠的危害和防护措施",
        "试剂盒开封后能存放多久",
        "多聚甲醛废液怎么处置",
    ],
    "notice": [
        "本周实验室安全培训的时间安排",
        "新发布的内部通知说了什么",
        "年度生物安全考核有哪些要求",
        "这次培训的重点内容是什么",
        "实验室管理制度最近有什么更新",
    ],
}

# 离线构建原型向量
role_centroids = {}
for role, queries in ROLE_PROTOTYPES.items():
    embeddings = encoder.encode(queries, normalize_embeddings=True)
    role_centroids[role] = np.mean(embeddings, axis=0)
    role_centroids[role] /= np.linalg.norm(role_centroids[role])


def route_query(query: str, top_k: int = 2, threshold: float = 0.5):
    """将 Query 路由到最相关的 Role"""
    q_emb = encoder.encode([query], normalize_embeddings=True)[0]

    scores = {}
    for role, centroid in role_centroids.items():
        scores[role] = float(np.dot(q_emb, centroid))

    ranked = sorted(scores.items(), key=lambda x: -x[1])

    results = []
    for role, score in ranked[:top_k]:
        if score >= threshold or not results:
            results.append({"role": role, "confidence": round(score, 4)})

    return results
```

**优势**：
- 仅需每个 Role 5~10 条手写示例即可启动，无需训练。
- 推理延迟 < 15ms（GPU）/ < 50ms（CPU），满足实时要求。
- 随时可通过增加原型样本来微调效果。

**不足**：
- 对近义 Role（如 sop vs knowledge）的区分能力有限。
- 不直接支持多标签输出（通过阈值间接实现）。

---

### 阶段二：成熟期（≥500 条标注数据）

**技术选型：BGE/BERT 微调多标签分类器**

当积累了足够的标注数据后，切换到微调分类模型作为主力。

**模型结构**：

```text
                    Query (tokenized)
                         │
                         ▼
              ┌─────────────────────┐
              │  bge-base-zh-v1.5   │   (frozen or fine-tuned, 768-dim)
              │  (Encoder Backbone)  │
              └─────────┬───────────┘
                        │ [CLS] pooling
                        ▼
              ┌─────────────────────┐
              │  Dropout(0.1)       │
              │  Linear(768 → 8)    │
              │  Sigmoid            │   ← 多标签，每个 Role 独立输出概率
              └─────────┬───────────┘
                        │
                        ▼
              [sop: 0.92, emergency: 0.03, regulation: 0.87, ...]
```

**训练策略**：

| 配置项 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| 骨干模型 | `BAAI/bge-base-zh-v1.5` | 与向量库 Encoder 共用，减少部署成本 |
| 损失函数 | `BCEWithLogitsLoss` | 多标签二元交叉熵 |
| 学习率 | 2e-5（骨干）/ 1e-3（分类头） | 差异学习率，骨干微调幅度小 |
| Batch Size | 32 | — |
| Epochs | 5~10 | Early stopping on val loss |
| 数据增强 | 同义改写 + 回译 + LLM 合成 | 解决样本不足问题 |

**标注数据获取策略**：

1. **LLM 批量合成**：用 GPT-4 / DeepSeek 根据每个 Role 的定义和典型问法，批量生成 200+ 条 Query，经人工审核后入库。
2. **线上日志挖掘**：将阶段一的 Prototype 匹配结果 + LLM 仲裁结果记录下来，高置信度的自动入库，低置信度的人工审核。
3. **主动学习**：定期从线上 Query 中挑选模型最不确定的样本（熵最高），交给标注人员。

**推理伪代码**：

```python
import torch
from transformers import AutoTokenizer, AutoModel

class IntentClassifier:
    def __init__(self, model_path: str, roles: list, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path).to(device)
        self.classifier = torch.nn.Linear(768, len(roles)).to(device)
        self.roles = roles
        self.device = device

    def predict(self, query: str, threshold: float = 0.4):
        inputs = self.tokenizer(query, return_tensors="pt",
                                max_length=64, truncation=True,
                                padding=True).to(self.device)
        with torch.no_grad():
            hidden = self.backbone(**inputs).last_hidden_state[:, 0, :]
            logits = self.classifier(hidden)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        results = []
        for role, prob in zip(self.roles, probs):
            if prob >= threshold:
                results.append({"role": role, "confidence": round(float(prob), 4)})

        results.sort(key=lambda x: -x["confidence"])

        if not results:
            best_idx = probs.argmax()
            results = [{"role": self.roles[best_idx],
                         "confidence": round(float(probs[best_idx]), 4)}]

        return results
```

---

### LLM 仲裁层（贯穿全阶段）

当 Stage 1 输出的 Top-1 置信度低于阈值（如 Prototype 方案 < 0.55，分类模型 < 0.6）时，触发 LLM 仲裁。

**Prompt 模板**：

```text
你是一个生物安全实验室知识库的意图识别助手。

用户提出了一个问题，请判断它属于以下哪个或哪几个知识类别，并给出理由。

## 类别定义
- sop: 指导具体操作流程（怎么做、步骤是什么）
- emergency: 突发事故应对处理（泄漏、暴露、针刺等怎么处置）
- regulation: 法规制度与合规要求（是否允许、需要满足什么要求）
- directory: 分类名录与清单查询（某病原体属于哪类、分级信息）
- knowledge: 概念原理与教学解释（原理是什么、为什么）
- equipment: 设备使用与故障排查（怎么校准、报警代码含义）
- reagent: 试剂耗材安全说明（怎么保存、MSDS 信息）
- notice: 通知公告与培训材料（培训要点、时间安排）

## 用户问题
{query}

## 输出要求
以 JSON 格式返回，包含 roles（数组）和 reason（字符串）。仅返回 JSON，不要其他内容。
示例: {"roles": ["emergency", "regulation"], "reason": "涉及事故处置流程和制度规定"}
```

**触发条件与成本控制**：

| 指标 | 预期值 |
| :--- | :--- |
| 触发率（阶段一 Prototype 方案） | ~15%~25% 的 Query |
| 触发率（阶段二微调模型） | ~3%~8% 的 Query |
| 单次调用延迟 | 200ms~800ms（取决于模型和并发） |
| 成本 | DeepSeek-V3 约 ¥0.001/次，可忽略 |

---

## 四、Query 预处理模块

Intent Router 之前应增加预处理管线，提升下游路由准确率。

### 1. 术语标准化

将用户口语表达映射到知识库中的标准术语。

```python
TERM_ALIASES = {
    "安全柜": "生物安全柜",
    "超净台": "超净工作台",
    "高压锅": "高压蒸汽灭菌器",
    "酒精": "乙醇",
    "甲醛": "多聚甲醛",
    "P2实验室": "BSL-2实验室",
    "P3实验室": "BSL-3实验室",
    "三级实验室": "BSL-3实验室",
    # ...持续维护
}

def normalize_terms(query: str) -> str:
    for alias, standard in TERM_ALIASES.items():
        query = query.replace(alias, standard)
    return query
```

### 2. Query 改写（可选，用 LLM 或规则）

对过短、模糊的 Query 进行补全改写，例如：

| 原始 Query | 改写后 |
| :--- | :--- |
| "针刺伤" | "实验室发生针刺伤后应该如何处理" |
| "BSL-3 要求" | "BSL-3 实验室需要满足什么管理要求和设施条件" |
| "MSDS" | "如何查看试剂的 MSDS 安全技术说明书" |

改写可通过简单的规则模板（短 Query + Role 线索 → 模板填充）在低延迟下实现，不一定需要调用 LLM。

---

## 五、多路召回融合策略

Intent Router 输出多个 Role 后，需要对各路召回结果进行融合。

### RRF（Reciprocal Rank Fusion）融合

```python
def rrf_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    ranked_lists: 多路召回结果，每路是按相关度排序的 chunk 列表
    k: RRF 常数，默认 60
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [{"chunk_id": cid, "rrf_score": score} for cid, score in fused]
```

### 完整检索管线

```text
Intent Router 输出: [regulation(0.91), emergency(0.72)]
        │
        ├── regulation 池: 向量召回 Top-20 + BM25 召回 Top-20 → RRF 融合 → Top-15
        │
        ├── emergency  池: 向量召回 Top-20 + BM25 召回 Top-20 → RRF 融合 → Top-15
        │
        ▼
    合并 30 条候选
        │
        ▼
    Cross-Encoder Re-rank (query, chunk) → Top-8
        │
        ▼
    Block Promotion 检查 → 最终上下文 (≤ 4000 tokens)
        │
        ▼
    LLM 生成回答
```

### Cross-Encoder 精排建议

| 配置 | 推荐 |
| :--- | :--- |
| 模型 | `BAAI/bge-reranker-v2-m3` 或 `maidalun1020/bce-reranker-base_v1` |
| 输入 | (query, chunk_content) pair |
| 输出 | 相关性分数 0~1 |
| 延迟 | 30 条候选约 50~80ms (GPU) |

---

## 六、评估与迭代

### 离线评估指标

| 指标 | 计算方式 | 目标 |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | 最高置信度 Role 命中正确标签 | ≥ 90% |
| **Top-2 Coverage** | 前两个 Role 中包含正确标签 | ≥ 97% |
| **Multi-label F1** | 多标签 Micro-F1 | ≥ 0.85 |
| **低置信度比例** | 需要 LLM 仲裁的 Query 占比 | ≤ 10% |

### 在线监控

- 记录每条 Query 的路由结果、置信度、是否触发 LLM 仲裁。
- 定期抽样人工审核路由准确率（周频，每次 100 条）。
- 监控 LLM 仲裁触发率，如持续上升说明分布漂移，需补充训练数据。

### 数据飞轮

```text
线上 Query 日志
      │
      ▼
  模型打标 + 置信度
      │
      ├── 高置信度 (≥ 0.85) ──→ 自动入库（伪标签）
      │
      ├── 中置信度 (0.5~0.85) ──→ LLM 复核 → 入库
      │
      └── 低置信度 (< 0.5) ──→ 人工标注 → 入库
      │
      ▼
  定期重训模型 (周频/月频)
```

---

## 七、技术选型总结

| 组件 | 冷启动期推荐 | 成熟期推荐 |
| :--- | :--- | :--- |
| **嵌入模型** | `BAAI/bge-base-zh-v1.5` | 同左（与向量库共用） |
| **Intent Router** | Prototype 原型匹配 | 微调多标签分类器 |
| **LLM 仲裁** | DeepSeek-V3 / Qwen2.5 | 同左（仅兜底，调用率 < 5%） |
| **Re-ranker** | `bge-reranker-v2-m3` | 同左 |
| **多路融合** | RRF (k=60) | 同左 |
| **框架** | Sentence-Transformers + PyTorch | 同左，可转 ONNX 加速 |

### 部署资源估算

| 场景 | GPU 需求 | CPU 可行性 |
| :--- | :--- | :--- |
| Prototype 匹配（bge-base 编码） | 不需要 | 可行，~50ms/query |
| 微调分类器推理 | 不需要 | 可行，~30ms/query |
| Cross-Encoder Re-rank（30 pairs） | 建议 | CPU ~300ms，可接受 |
| LLM 仲裁 | API 调用 | N/A |

---

## 八、实施路线图

```text
Week 1-2:  手写每 Role 20 条原型 Query → 部署 Prototype 方案上线
           同步: 配置 LLM 仲裁 Prompt，对接日志记录

Week 3-4:  用 LLM 批量合成训练数据 (每 Role 100+条)
           人工审核 + 清洗 → 初版标注集 (≥500 条)

Week 5-6:  微调 bge-base 分类器，离线评估
           A/B 测试: Prototype vs 微调模型

Week 7-8:  上线微调模型，Prototype 降级为 fallback
           部署 Cross-Encoder Re-ranker
           启动数据飞轮

持续:      每月从日志中补充标注数据，重训模型
           监控路由准确率和 LLM 仲裁触发率
```
