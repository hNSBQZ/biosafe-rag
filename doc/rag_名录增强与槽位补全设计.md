# RAG 名录增强与槽位补全（Query Understanding）设计

## 背景与目标

现有总体链路（`readme.md` 43–77 行）在进入 `[RAG 检索管线]` 前缺少一层**面向检索的 Query 理解**，导致以下典型问题：

- **问题 1：术语简称/别名影响召回**  
  用户口语里可能说“新冠/猴痘/腮腺炎/甲流”，而知识库与名录里用的是更正式的名称（如“新型冠状病毒”“Mpox virus”等），若不做归一化会降低 BM25 与向量召回。
- **问题 2：跨文档依赖导致检索缺槽**  
  名录表格（如 `doc/catolog.md`）给出“病毒 → 实验室等级（BSL/ABSL）”，而“温度设定/培养条件/操作条款”通常写在**对应 BSL 等级的制度/条款**里。若检索 Query 不带实验室等级（或不带活动类型），就会“召回到相关但不够近”的片段。
- **问题 3：希望同时产出 Dense 与 BM25 的更细策略**  
  既然已经维护了大量关键词与结构信息，就应该把它用在**意图识别 + 槽位补全 + 双查询生成**上，而不是只把原始 query 扔进检索。

本文档提出一个可插入到现有架构中的模块：**Query Understanding（名录增强与槽位补全）**，其产出是：

- **结构化槽位**：`entity(病原体)`、`activity(实验活动)`、`lab_level(BSL/ABSL)`、`roles(检索域)`、`filters(metadata 过滤)`  
- **双检索语句**：`dense_query`（语义向量友好）与 `bm25_query`（关键词/短语友好）  
- **检索计划**：是否先查名录、查哪些 Role、是否需要分段召回与融合

> 说明：本文只设计“怎么检索得更准”，不提供任何具体病原体“培养/扩增/温度参数”的操作性建议。

---

## 总体位置：插入到现有流程的哪里

建议把该模块放在 **“LLM 输出需要的 Role + 改写后的检索 Query”** 与 **“RAG 检索管线”** 之间（也可以在“LLM 首轮之前”先做轻量归一化，见后文的两种集成方式）。

概念流程如下：

```text
User Query
  │
  ├─ FuncCall 规则命中 → 返回 XR JSON
  │
  └─ 进入知识问答路径
       │
       ├─（可选）轻量归一化：分词/纠错/同义词
       │
       ├─ LLM 第一轮（判断能否直答 / 给出 NEED_RAG+role）
       │
       └─ 若 NEED_RAG：
            ▼
        Query Understanding（本模块）
            - 病原体实体链接（名录/别名）
            - 活动类型识别（培养/动物感染/样本检测/灭活等）
            - 实验室等级补全（BSL/ABSL）
            - 生成 dense_query + bm25_query + filters
            ▼
        Hybrid Retrieval（向量 + BM25 + 融合/重排）
            ▼
        LLM 第二轮（用召回材料回答）
```

---

## 数据准备（离线）：把名录表格变成“可补全槽位”的知识

### 1）名录结构化（Catalog Record）

把 HTML 表格解析为结构化记录（示例字段，实际以你解析器输出为准）：

```json
{
  "entity_id": "virus:mpox",
  "cn_name": "猴痘病毒",
  "en_name": "Mpox virus",
  "taxonomy": "痘病毒科",
  "hazard_class": "第一类",
  "lab_level": {
    "virus_culture": "BSL-3",
    "animal_infection": "ABSL-3",
    "uncultured_material": "BSL-2",
    "inactivated_material": "BSL-2",
    "noninfectious_material": "BSL-1"
  },
  "transport": { "category": "A", "un": "UN2814" },
  "notes": ""
}
```

关键点：

- **实体主键**：建议引入稳定的 `entity_id`（不要只靠中文名），便于跨表/跨文档引用。
- **活动类型字段**：名录中“病毒培养a / 动物感染实验b / …”是天然的“活动槽位”来源，后续用于把用户问题映射到正确的 BSL/ABSL 值。

### 2）别名与简称词典（Alias Lexicon）

为每个实体建立别名集合，用于 Query 实体链接与检索扩写。

别名来源建议分三层（从确定性到弱确定性）：

- **强别名（规则生成）**：
  - 去掉“病毒/病原体/冠状病毒”等后缀得到简称：`猴痘`、`新冠`（可维护规则白名单避免过度泛化）
  - 中英文括号展开：`新型冠状病毒(SARS-CoV-2)` → 两者互为别名
  - 常见缩写/变体：`SARS2`、`SARS CoV 2`、`CoV-2` 等
- **中别名（人工维护）**：
  - 领域常用叫法：`甲流`、`乙流`、`诺如` 等
  - 旧称/别称（如文献中常见的历史命名）
- **弱别名（向量/拼写相似）**：
  - 针对 ASR 错字/近音字，做轻量 fuzzy（编辑距离/拼音相似）候选召回，再用打分规则选 Top-1。

建议输出统一结构：

```json
{
  "alias": "新冠",
  "entity_id": "virus:sars-cov-2",
  "strength": "medium"
}
```

### 3）实体索引（Entity Linker Index）

在线实体链接建议同时具备两类索引：

- **字典/AC 自动机索引**：覆盖强别名，速度极快、可解释。
- **Embedding 索引（可选）**：用于“没匹配到字典，但看起来像某个实体名”的兜底候选；可以只对“别名短语”做向量化，规模通常很小。

---

## 在线：Query Understanding 的核心输出

### 设计成一个纯函数：输入 query，输出“检索计划”

```json
{
  "original_query": "猴痘怎么培育？温度怎么设？",
  "normalized_query": "猴痘 怎么 培育 温度 怎么 设定",
  "intent": {
    "primary": "ask_parameter",
    "secondary": ["ask_procedure"],
    "confidence": 0.78
  },
  "slots": {
    "entity": {
      "entity_id": "virus:mpox",
      "matched_alias": "猴痘",
      "canonical": "猴痘病毒 (Mpox virus)",
      "confidence": 0.91
    },
    "activity": {
      "type": "virus_culture",
      "confidence": 0.63
    },
    "lab_level": {
      "value": "BSL-3",
      "source": "catalog",
      "confidence": 0.95
    }
  },
  "retrieval": {
    "roles": ["directory", "regulation", "sop"],
    "filters": { "lab_level": ["BSL-3"], "entity_id": ["virus:mpox"] },
    "dense_query": "猴痘病毒 Mpox virus 病毒培养 BSL-3 实验室 温度 设置 培养 条款 要求",
    "bm25_query": "\"猴痘病毒\" OR 猴痘 OR \"Mpox virus\" BSL-3 病毒培养 温度 设定 要求 条款",
    "need_catalog_first": true
  }
}
```

其中每个字段的作用：

- **intent**：决定“是否要补全 lab_level/活动槽位”、优先检索哪些 Role
- **entity**：决定别名扩写、过滤条件（如 `entity_id`）与优先召回的名录 chunk
- **activity**：把问题落到名录的某一列（如 `virus_culture` / `uncultured_material`）
- **lab_level**：为后续检索加入强约束（过滤/查询扩写）
- **dense_query / bm25_query**：分别优化向量召回与倒排召回

---

## 意图识别：为了“补哪些槽位”

这里的“意图”不需要做成很细的 N 类分类，更实用的是：识别**需要补全哪些检索要素**。

### 1）参数/条件类（Parameter Seeking）

触发信号（示例）：`温度/湿度/时间/转速/压力/浓度/条件/设置/参数/范围/阈值`

- **检索目标**：通常落在 `regulation / sop / equipment` 等 Role
- **强依赖槽位**：`lab_level`（对应 BSL 条款），`activity`（影响 BSL）

### 2）流程/怎么做类（Procedure Seeking）

触发信号（示例）：`如何/怎么/步骤/流程/先后/注意事项/培养/扩增/分离/操作`

- **检索目标**：`sop` 为主，辅以 `regulation`（合规前置条件）
- **强依赖槽位**：`entity`、有时需要 `activity`

### 3）合规/是否允许类（Compliance Seeking）

触发信号（示例）：`能不能/是否允许/合规/审批/许可/要求/不得/禁止`

- **检索目标**：`regulation` 为主 + `directory`（风险分类与等级依据）
- **强依赖槽位**：`entity`、`hazard_class`、`lab_level`

### 4）名录/分类查询（Directory Seeking）

触发信号（示例）：`属于几类/BSL 几级/ABSL/UN 编号/运输包装/目录/名录/危害分类`

- **检索目标**：`directory` 为主
- **强依赖槽位**：`entity`

> 实现上：意图识别可以先用规则打分（类似你已有 `score_role` 思路），低置信度再交给 LLM（在线成本可控）。

---

## 槽位补全：实体 → 活动 → 实验室等级

### 1）实体槽位：病原体实体链接（Entity Linking）

目标：把用户提到的“口语词/简称”链接到名录中的“规范实体”。

推荐策略（按优先级）：

- **精确命中强别名**：字典/AC 扫描，得到候选实体（可返回多个）
- **冲突消解**：若多个候选同时命中，用下列信号打分选 Top-1：
  - 命中别名的强度（strong > medium > weak）
  - 与 query 其他词的共现（如“UN2814/BSL-3/痘病毒科”等）
  -（可选）alias embedding 与 query embedding 相似度
- **兜底**：仍不确定时，进入“多实体并行检索”（Top-2/Top-3）或让 LLM 在第二轮回答时做澄清提问（对话型产品更自然）。

### 2）活动槽位：把用户问题映射到名录列（Activity Mapping）

名录里同一实体有多种实验活动对应不同 BSL/ABSL，用户问题里常见触发词可以映射到活动类型：

- `培养/分离/扩增/滴定/活病毒培养物` → `virus_culture`
- `动物感染/攻毒/动物实验` → `animal_infection`
- `样本/未经培养/临床材料/感染材料/核酸检测(未灭活)` → `uncultured_material`
- `灭活/固定/裂解` → `inactivated_material`
- `无感染性/DNA/cDNA` → `noninfectious_material`

活动槽位的价值在于：**它是 lab_level 的选择器**。没有它，就只能拿一个默认列（会误导）。

### 3）实验室等级槽位：从名录补全（Lab Level Fill）

规则：

1. 若已链接 `entity` 且已识别 `activity`：直接从名录记录取 `lab_level[activity]`  
2. 若只有 `entity` 没有 `activity`：
   - 根据意图做默认：如“培养/培育”默认 `virus_culture`；“样本检测”默认 `uncultured_material`
   - 或返回多个可能 lab_level，并在检索中并行尝试（更鲁棒，代价是多检索一次）
3. 若 entity 也未确定：跳过 lab_level 过滤，只做通用检索 + 引导澄清

---

## 双查询生成：Dense Query 与 BM25 Query 各司其职

你提出“既适合 dense 也适合 BM25”，建议明确两条 query 的设计目标：

### 1）Dense Query：语义向量更友好

原则：

- 使用**自然语言 + 关键约束**，避免过多布尔/符号
- 把补全槽位显式写进去（尤其是 `lab_level`、`activity`）
- 把规范实体名与英文名都放进去（对跨语种 embedding 更稳）

模板（示例）：

```text
{canonical_cn} {canonical_en} {activity_cn} {lab_level}
用户问题：{original_query}
需要的内容：{intent_hint}
```

### 2）BM25 Query：倒排检索更友好

原则：

- 以**高 IDF 关键词**为主（实体名、BSL、条款/制度关键短语）
- 适度引入短语查询与 OR 扩写（视你的 BM25 实现支持度）
- 控制长度，避免噪声词稀释（尤其是口语停用词）

模板（示例）：

```text
("{canonical_cn}" OR {alias1} OR "{canonical_en}") {lab_level} {activity_kw}
温度 OR 参数 OR 设置 OR 条款 OR 要求
```

---

## 检索策略：Hybrid + 融合 +（可选）分段召回

### 1）建议的检索执行顺序

当检测到“病原体实体 + 参数/合规/等级相关”时，推荐一个两段式策略：

- **段 1：名录优先（Directory-first）**  
  先把与实体相关的名录记录 chunk 召回（或直接从结构化 record 生成一条“实体卡片 chunk”），用于：
  - 为段 2 检索补全 `lab_level/hazard_class`
  - 在最终上下文中提供“依据来源”（减少 LLM 编造）
- **段 2：按槽位定向检索（Slot-conditioned retrieval）**  
  用 `dense_query + bm25_query` 在目标 roles（如 `regulation/sop`）中检索，并加上 metadata 过滤（如 `lab_level=BSL-3`）。

### 2）融合方式（向量 + BM25）

推荐使用可解释且工程成本低的融合：

- **RRF（Reciprocal Rank Fusion）**：对向量 Top-k 与 BM25 Top-k 做排名融合，鲁棒性强
- **轻量重排（可选）**：用 cross-encoder 或小模型对融合后的 Top-N 进行重排（你在 `readme.md` 的“后续可优化方向”里已经提到）

### 3）过滤条件（Metadata Filter）的使用

当 `lab_level` 补全成功时，过滤优先级应高于 query 文本扩写：

- **filters**：`role in {...}`、`lab_level in {...}`、（可选）`entity_id in {...}`  
  过滤能显著降低“相关但不适用”的片段进入候选集。

> 注意：filter 过严会导致召回为空。建议设置“软硬两级”：先硬过滤检索一次；若空，再放宽为仅 role 过滤 + query 扩写。

---

## 两种集成方式：LLM 主导 vs 检索主导

### 方案 A：LLM 首轮仍产出 role，但 Query Understanding 决定最终检索 query

优点：保持你现有“LLM 判断能否直答”的优势；适合前缀缓存的在线形态。  
做法：

- LLM 首轮输出 `[NEED_RAG:role1,role2]` + `QUERY:...`
- Query Understanding：
  - 用实体链接 + 名录补全覆盖/修正 LLM 的 query（尤其补充 `lab_level/activity`）
  - 生成 `dense_query/bm25_query/filters`

### 方案 B：Query Understanding 先行，LLM 首轮只做“能否直答 + 意图”

优点：更稳定、可解释；对弱模型更友好。  
做法：

- Query Understanding 先把实体/实验室等级补出来
- LLM 首轮只需要判断“prefill 能否回答”，以及选择 roles（可以更短、更结构化）

实践建议：先落地 **方案 A**（改动最小），等稳定后再考虑 B。

---

## 评估与验收（建议）

为了确认这套“名录增强 + 槽位补全 + 双查询”真的提升效果，建议至少做三类离线回放评测：

- **实体链接准确率**：简称/别名 query 的 Top-1 entity 命中率
- **带槽检索提升**：同一批 query，对比“原始 query 检索” vs “槽位补全检索”的 Recall@k / MRR
- **空召回率**：过严 filter 导致的空结果比例（需要优化软硬两级）

---

## 建议你下一步补充的最小实现清单

如果以“最小可用”为目标，工程上可以按这个顺序实现：

1. **名录结构化 JSON**（从 `doc/catolog.md` 解析出 records）
2. **alias_map（强别名）**（规则生成 + 少量人工补丁）
3. **entity_linker（字典命中 + 冲突消解）**
4. **activity_mapper（关键词映射）**
5. **lab_level_fill（从 record 取值）**
6. **query_builder（dense_query + bm25_query + filters）**
7. **hybrid_search + RRF 融合**

