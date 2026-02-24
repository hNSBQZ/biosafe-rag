# 文档分块与RAG检索策略设计说明

本文档介绍了一种专为结构化长文本（如生物安全规范、应急预案、操作指南等）设计的 **“文档 -> Block（语义块） -> Chunk（检索块）”** 自顶向下两级分块算法，以及基于该分块结构的 **意图路由与多路回归 RAG 检索策略**。

## 一、 算法概述与核心伪代码

该算法旨在解决传统文本切分工具在处理具有复杂层级结构的 Markdown 文档时，容易破坏语义完整性、丢失上下文的问题。

### 核心处理流程伪代码

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

---

## 二、 与普通递归分块（Recursive Chunking）的区别与优势

在 RAG（检索增强生成）应用中，常用的普通递归分块（例如 LangChain 的 `RecursiveCharacterTextSplitter`）通常基于预设的字符分隔符（如 `["\n\n", "\n", " ", ""]`）和固定的 Token 长度进行纯文本切片。

与普通递归分块相比，本 Semantic Block-Chunking 算法在结构化文档处理上具有显著优势：

| 特性对比 | 普通递归分块 (Recursive Chunking) | 本算法 (Semantic Block-Chunking) |
| :--- | :--- | :--- |
| **切分依据** | 纯粹基于长度和换行符，属于“物理切分” | 基于文档本身的逻辑结构（Markdown 标题、中文层级、数字编号），属于“语义切分” |
| **上下文感知能力** | 极弱。切断后丢失上下文，大模型不知道该段落属于哪一章哪一节 | 极强。维护全局树状状态，每个 Chunk 强制注入其在文档树中的**完整路径（面包屑）** |
| **碎片控制** | 容易切出只有半句话或一两行的无意义 Chunk | 具有完善的“过小合并”机制，杜绝逻辑碎片 |
| **层级嵌套处理** | 扁平化处理，不区分父标题和子内容，标题可能与正文分离 | 具有父标题（空壳）合并机制，保留父子从属关系，确保标题永远引领其正文 |

### 核心优势详解

1. **彻底解决 RAG 检索中的“上下文丢失”痛点**
   普通算法搜到一句话“*操作时必须穿戴防护服*”，大模型无法判断这是适用于 BSL-1 还是 BSL-3 实验室。本算法通过在 Chunk 首部打上 `第三章 实验室要求 > 3.2 BSL-3实验室` 的前缀，完美保留了全局语境，大幅提升向量检索的召回准确率。
2. **保证具体条款的语义完整性**
   利用“短列表项不切分、长编号条目才切分”的策略。例如，遇到 `1. 试管; 2. 离心机` 这种短枚举，算法会判定它们属于同一个语义段而拒绝切分；只有遇到 `1. 当实验室发生严重泄漏时，应当采取以下措施...` 这种长文本条目，才会将其作为独立的检索单元。
3. **消除“悬空标题”**
   普通基于长度的算法经常把标题切在一个 Chunk 的末尾，而将真正的正文切在下一个 Chunk 的开头。本算法严格以标题为边界，确保标题和它所管辖的正文始终绑定在一起。

---

## 三、 核心算法特色点解析

### 1. 面包屑路径的追踪与注入 (Breadcrumb Injection)

*   **原理**：在逐行解析文档时，算法内部维护了一个**标题层级栈**（Level Stack）。
*   **应用**：在最终生成最底层的 Chunk 时，将这个栈的路径拼接为字符串，**作为前缀强制注入到该 Chunk 内容的首部**。
*   **价值**：极大提升了向量数据库的召回效果，使得最底层的细节操作也能因为包含了外层路径信息而被精准命中。

### 2. “空壳标题”的智能合并 (Shell Block Merging)

*   **痛点场景**：如“# 五、应急处理程序”下紧跟“## （一）总则”，空壳标题会成为废块。
*   **算法特色**：自动识别空壳标题，**将其合并到下一个子 Block 的面包屑路径中，并销毁自身实体**。

### 3. “过大拆分与过小合并”的自适应弹性呼吸机制 (Adaptive Sizing)

*   **Block 级动态降维拆分**：寻找超大块内部最浅层级的子标题作为边界进行降维分解。
*   **Chunk 级“三步约束” (Merge -> Split -> Merge)**：先合并过小片段，再针对无子标题的超长段落进行间距硬拆分，最后清理拆分产生的碎片。

### 4. 复杂表格的结构化“解构与重组” (Structured Table Parsing)

*   将 HTML 表格的 `rowspan/colspan` 规范化，智能聚合左侧表头与数据，重构为 `字段名: 字段值` 的 Record-level Chunk，并注入面包屑。

---

## 四、 RAG 检索策略设计思路

为配合上述分块算法，RAG 系统在检索层引入了基于意图路由（Intent Routing）和多路回归的精细化策略。

### 1. 查询处理与多路回归
- **前置处理**：用户的 Query 会先经过**术语补全**、**信息补全**和**意图识别**。
- **意图路由策略**：根据识别出的意图，决定多路召回策略。例如，一个问题可能同时需要“法规制度”和“教学解释”两种维度的知识。
- **定向降噪**：系统根据识别出的目标角色（Role），紧缩并在对应 Role 的 Chunk 池中进行定向检索。这样可以屏蔽无关领域的干扰，大幅增加召回内容的有效信息比例。

### 2. 块级升格机制 (Block Promotion)
- **局部召回**：首先通过向量或 BM25 召回高相关度的 Chunk。
- **一致性校验与升格**：检查召回的 Chunk 所属的 Block 的 Role 是否与该 Chunk 的 Role 一致。
- **完整上下文补全**：如果一致，则认为整个 Block 都是高度相关的同质内容，此时**将整个 Block 召回（升格）**。这使得输入给 LLM 的上下文更加全面和完整，从而生成质量更高的回答。

### 3. Chunk 与 Block 的 Role 打标逻辑

为了实现上述意图路由，在入库时需要通过规则加权与大模型辅助，为每一个 Chunk 和 Block 打上相应的 Role 标签。

```python
ROLES = ["sop", "emergency", "regulation", "directory", "knowledge", "equipment", "reagent", "notice"]

LEXICON[role] = set(keywords...)          # 该角色下的触发关键词词表
BOOST_TITLE[role] = set(title_keywords...)# 标题命中加权词表

def score_role(text, title):
    scores = {role: 0 for role in ROLES}

    for role in ROLES:
        # keyword hits (内容关键词命中)
        for kw in LEXICON[role]:
            if kw in text:
                scores[role] += 1

        # title boosts (block/chunk所在标题路径命中，权重更高)
        for tk in BOOST_TITLE[role]:
            if tk in title:
                scores[role] += 3

    # structure boosts (特定的格式结构特征加分)
    if looks_like_definition(text):         # 例如："2.1 XXX ...", 或者包含 "术语/定义"
        scores["knowledge"] += 4

    if starts_like_steps(text):             # 例如开头是："步骤", "Step", "（1）", "一、", "1."
        scores["sop"] += 3

    if contains_any(text, ["必须","不得","禁止","应当","要求","规定"]):
        scores["regulation"] += 2

    if contains_any(text, ["泄漏","暴露","针刺","应急","处置","事故"]):
        scores["emergency"] += 4

    if contains_any(text, ["报警","故障","维护","保养","校准","代码"]):
        scores["equipment"] += 4

    if contains_any(text, ["SDS","MSDS","安全技术说明书","危害","急救措施","灭火措施"]):
        scores["reagent"] += 5

    if contains_any(text, ["目录","名录","清单","分类","分级","危害等级","实验室等级"]):
        scores["directory"] += 4

    return scores

def infer_chunk_role(chunk_text, heading_path):
    scores = score_role(chunk_text, heading_path)
    best_role, best_score = argmax(scores)

    # tie-break (保持简单与稳定)
    # 优先级设定：emergency > reagent > equipment > directory > sop > regulation > knowledge > notice
    if has_tie(scores):
        best_role = tie_break_by_priority(scores)

    # 计算置信度 (Confidence)
    second_score = second_max(scores)
    confidence = best_score - second_score   # 与第二名的分差
    if best_score == 0:
        best_role = "knowledge"              # 兜底选项（也可选 regulation，视业务场景而定）
        confidence = 0

    # 如果置信度过低，在此处可选择请求大模型（LLM）进行精准判断
    
    return best_role, confidence

def infer_block_role(block, chunks):
    # chunks 是已经分配了 chunk_role 和 confidence 的该 block 下属的所有子节点
    votes = {role: 0 for role in ROLES}

    for c in chunks:
        if c.block_id == block.id:
            # 用置信度作为投票加权权重
            weight = 1 + max(0, c.confidence)
            votes[c.role] += weight

    best_role = argmax(votes)

    # 如果 block 的标题强触发某类，可在此叠加标题修正逻辑
    if title_contains(block.title, "术语") or title_contains(block.title, "定义"):
        best_role = "knowledge"
    if title_contains(block.title, "应急") or title_contains(block.title, "事故"):
        best_role = "emergency"

    return best_role
```

### 4. 角色 (Role) 定义与典型问法

目前系统设定以下 8 个主要的 Role：

| Role 分类 | 定义说明 | 包含的文档/内容类型 | 典型用户问法 |
| :--- | :--- | :--- | :--- |
| **sop** <br>*(SOP/实验步骤类)* | 指导具体操作流程的说明性内容 | 教学教案、实验指导书、标准操作步骤、流程说明 | 怎么做、步骤是什么、先后顺序是什么、怎么配制、怎么设置 |
| **emergency** <br>*(应急预案/安全处置类)* | 突发状况下的应对与处理方案 | 泄漏、暴露、针刺、火灾、停电等事故处置流程 | 如果发生 X 怎么处理、怎么处置、怎么上报/隔离/清理 |
| **regulation** <br>*(法规制度/合规条款类)* | 必须遵守的硬性规定与管理要求 | 生物安全通用要求、制度要求、废弃物处置规定、BSL/台账记录要求等 | 是否允许、需要满足什么要求、记录怎么做、对应什么实验室等级或管理要求 |
| **directory** <br>*(名录/目录/清单类)* | 结构化的分类分级清单与参考表 | 病原体名录、分类目录、分级信息、结构化表格或清单 | 某病原体属于哪类、危害或分级是什么、对应实验室等级要求是什么、英文名/别名是什么 |
| **knowledge** <br>*(教学解释/原理知识类)* | 解释概念、背景及原理的知识性内容 | 课程讲义、教材节选、FAQ、原理说明、概念解释 | 原理是什么、为什么这么做、该术语是什么意思、相关的背景知识 |
| **equipment** <br>*(设备说明/维护手册类)* | 仪器的使用、故障排查与维护保养 | 离心机、超净台、生物安全柜、PCR 仪说明书、报警代码表、维护保养记录 | 设备怎么设置/校准、报警代码是什么意思、怎么维护、使用注意事项 |
| **reagent** <br>*(试剂/耗材说明类)* | 化学品或生物耗材的特性与使用安全说明 | MSDS/SDS、安全技术说明书、试剂盒说明书、储存及运输条件说明 | 怎么保存、有效期多长、有什么危害及如何防护、废弃后如何处置、兼容性如何 |
| **notice** <br>*(通知/公告/培训材料类)* | 组织内部下发的宣发或时效性内容 | 培训课件 PPT、内部公告、课程通知、考核要求 | 培训的要点是什么、通知的具体内容、时间安排是怎样的、内部规定有何更新 |
