# 在线服务模块架构设计

## 一、现有代码盘点

| 文件 | 职责 | 定位 |
|:---|:---|:---|
| `funcall_detector.py` | FuncCall 关键词规则识别 | **在线**，已完成 |
| `split_chunk.py` | 文档分块 | 离线 |
| `process_table.py` | HTML 表格解析 → `table.json` | 离线 |
| `chunk_processor.py` | Chunk Role 打标（关键词 + LLM 兜底） | 离线 |
| `llm_client.py` | LLM API 封装（同步 / Batch） | 通用基础设施 |
| `milvus_manager.py` | Milvus 向量库 CRUD | 通用基础设施 |
| `config.py` | 配置管理 | 通用基础设施 |
| `pipeline.py` | 离线处理编排（分块→打标→embedding→导出） | 离线 |

---

## 二、在线链路新增模块

FuncCall 未命中后，进入在线服务链路。拆为 **5 个新文件**，每个职责单一、可独立测试：

```text
query_handler.py            ← 薄编排层，串联所有模块
     │
     ├── funcall_detector.py       (已有，无状态函数)
     │
     ├── slot_parser.py            ← NEW: LLM 输出解析 (纯函数，无状态)
     │
     ├── catalog_manager.py        ← NEW: 名录查表 + Query 增强 (有状态，持有内存索引)
     │
     ├── experiment_store.py       ← NEW: 实验提示词管理 (有状态，加载 YAML)
     │
     └── retriever.py              ← NEW: RAG 检索封装 (有状态，依赖 milvus_manager)
```

---

## 三、各模块详细设计

### 3.1 `slot_parser.py` — LLM 输出解析器

- **形式**：纯函数模块，无状态、无外部依赖
- **为什么单独拆**：这是整个在线链路的"信息枢纽"——LLM 第一轮输出经过它变成结构化的 `LLMSlots`，后续所有模块（名录查表、Query 增强、RAG 检索）都依赖它的输出。单独拆出来可以用大量 case 直接单测 `parse_llm_output()`，不需要真调 LLM

```python
@dataclass
class LLMSlots:
    needs_rag: bool
    answer: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    pathogen: Optional[str] = None
    activity: Optional[str] = None
    equipment: Optional[str] = None
    query: Optional[str] = None
    hedging: bool = False

def parse_llm_output(output: str, original_query: str) -> LLMSlots:
    """解析 LLM 第一轮输出，提取 [NEED_RAG] 标记和各槽位"""
    ...

def keyword_role_classify(query: str, top_k: int = 2) -> List[str]:
    """关键词规则兜底 Role 分类（复用 chunk_processor.score_role）"""
    ...
```

**模块间依赖**：`keyword_role_classify` 调用 `chunk_processor.score_role()`（已有的便捷函数），是唯一的跨模块依赖。

---

### 3.2 `catalog_manager.py` — 名录管理与 Query 增强

- **形式**：有状态的类
- **为什么用类**：启动时一次性加载 `table.json` 并建索引（中文名、英文名反向映射），之后每次请求复用
- **职责边界**：只关心"给我一个病原体名 + 活动类型，返回 BSL 等级并增强 query"，不关心 LLM 怎么调、检索怎么做

```python
@dataclass
class CatalogRecord:
    cn_name: str
    en_name: str
    hazard_class: str
    lab_level: Dict[str, str]     # activity → BSL/ABSL 等级
    transport: Optional[Dict] = None
    notes: str = ""

class CatalogManager:
    """名录查表 + Query 增强，启动时加载一次"""

    def __init__(self, table_json_path: str = "table.json"):
        self._catalog: Dict[str, CatalogRecord] = {}
        self._load(table_json_path)

    def fuzzy_lookup(self, name: str) -> Optional[CatalogRecord]:
        """三级 fallback 模糊匹配：精确 → 包含 → 相似度"""
        ...

    def resolve_lab_level(self, record: CatalogRecord,
                          activity: Optional[str]) -> Optional[str]:
        """从 CatalogRecord 提取 BSL 等级"""
        ...

    def enhance_query(self, slots: LLMSlots) -> str:
        """根据槽位做名录查表 + BSL 注入，返回增强后的检索 query"""
        ...
```

#### `table.json` → `CatalogRecord` 的列名映射

四张表的活动列名不同，需统一映射到标准 activity key：

```python
ACTIVITY_COLUMN_MAP = {
    # 表1 病毒
    "病毒培养":               "culture",
    "动物感染实验":           "animal",
    "未经培养的感染材料的操作": "sample",
    "灭活材料的操作":         "inactivated",
    "无感染性材料的操作":     "noninfectious",
    # 表2/3 细菌、真菌
    "活菌操作":               "culture",
    "样本检测":               "sample",
    "非感染性材料的实验":     "noninfectious",
    # 附录 朊病毒
    "组织培养":               "culture",
    "动物感染":               "animal",
    "感染性材料的检测":       "sample",
}
```

#### 三级 fallback 模糊匹配逻辑

1. **精确匹配**：中文名或英文名完全一致
2. **包含匹配**：处理 LLM 输出少了/多了修饰词的情况（"XX杆菌" vs "XX芽孢杆菌"），取最长匹配
3. **字符串相似度**：`SequenceMatcher.ratio() >= 0.6`，处理错字/异体字

---

### 3.3 `experiment_store.py` — 实验提示词管理

- **形式**：有状态的类
- **为什么用类**：启动时加载所有实验 YAML 到内存，按 ID 查找并填充 prompt 模版

```python
@dataclass
class Experiment:
    id: str
    name: str
    knowledge: str              # ~2000 字实验场景知识
    description: str = ""

class ExperimentStore:
    """实验提示词管理"""

    def __init__(self, experiments_dir: str = "experiments",
                 prompt_template: str = SYSTEM_PROMPT_TEMPLATE):
        self._experiments: Dict[str, Experiment] = {}
        self._prompt_template = prompt_template
        self._load_all(Path(experiments_dir))

    def get_experiment(self, experiment_id: str) -> Experiment:
        ...

    def build_system_prompt(self, experiment_id: str) -> str:
        """取出实验知识，填充到 prompt 模版，返回完整 system prompt"""
        ...

    def build_messages(self, experiment_id: str, query: str,
                       rag_context: Optional[str] = None) -> List[Dict]:
        """构建完整 messages 列表（system + user）
        
        system prompt 固定不变（命中前缀缓存），
        RAG 上下文（每次不同）作为 user message 追加。
        """
        ...
```

#### 实验知识存储格式：YAML

每个实验一个文件，放 `experiments/` 目录：

```text
experiments/
├── pcr_amplification.yaml
├── nucleic_acid_electrophoresis.yaml
├── gram_staining.yaml
└── ...
```

单个文件结构：

```yaml
# experiments/pcr_amplification.yaml
id: pcr_amplification
name: PCR扩增实验
description: 聚合酶链式反应扩增目标DNA片段

knowledge: |
  本实验为 PCR（聚合酶链式反应）扩增实验，目的是通过体外酶促反应
  扩增特定 DNA 片段。

  ## 实验材料
  - 模板 DNA（浓度 10-100 ng/μL）
  - 正向引物和反向引物（各 10 μM）
  - Taq DNA 聚合酶（5 U/μL）
  ...

  ## 操作步骤
  1. 在冰上配制 PCR 反应体系（总体积 50 μL）
  ...

  ## 注意事项
  - 所有操作在超净工作台内完成
  - 使用含滤芯吸头防止气溶胶污染
  ...
```

**为什么选 YAML**：

| 对比维度 | JSON | YAML | Python dict |
|:---|:---|:---|:---|
| 多行文本编辑体验 | 差（需 `\n` 转义） | **好**（`\|` 保留换行） | 差（三引号但在代码里） |
| 结构化 metadata | 支持 | 支持 | 支持 |
| 改知识要不要改代码 | 不用 | 不用 | **要** |
| 额外依赖 | 无 | `pyyaml` | 无 |

#### 与前缀缓存的衔接

`build_system_prompt()` 返回填充好的完整 system prompt。WebSocket 连接建立时：

```text
前端通知切换到实验 "pcr_amplification"
    → experiment_store.build_system_prompt("pcr_amplification")
    → 得到填充好的 system prompt
    → 发给 SGLang 预热 KV Cache
```

后续请求中 `build_messages()` 构建的 messages 里 system prompt 不变，自动命中前缀缓存。

---

### 3.4 `retriever.py` — RAG 检索封装

- **形式**：有状态的类
- **为什么单独拆**：检索策略（Dense/BM25 权重、top_k 分配、结果融合）是独立关注点，后续优化空间大（加 Cross-Encoder 精排、调权重比等），需要频繁修改和实验

```python
class Retriever:
    """Hybrid RAG 检索（Dense + BM25）"""

    def __init__(self, milvus_manager: MilvusManager,
                 embedding_client,
                 dense_weight: float = 0.6,
                 bm25_weight: float = 0.4):
        self._milvus = milvus_manager
        self._emb_client = embedding_client
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

    def search(self, query: str, roles: List[str],
               top_k: int = 8) -> List[Dict]:
        """Hybrid 检索 + 按 roles 过滤 + 结果融合去重
        
        对每个 role 分别检索再合并，确保各 role 都有召回。
        """
        ...

    def _dense_search(self, query: str, role: str, top_k: int) -> List[Dict]:
        """向量相似度检索"""
        ...

    def _bm25_search(self, query: str, role: str, top_k: int) -> List[Dict]:
        """BM25 稀疏检索（匹配面包屑中的 BSL 等级等关键词）"""
        ...

    def _merge_results(self, dense_hits, bm25_hits) -> List[Dict]:
        """RRF 或加权分数融合"""
        ...
```

---

### 3.5 `query_handler.py` — 在线请求主控制器

- **形式**：类（**薄编排层**，不含业务逻辑，只负责调用各模块并传递数据）
- **所有依赖通过构造函数注入**，自身不创建任何子模块

```python
class QueryHandler:
    """在线查询主控制器"""

    def __init__(self, llm_client: LLMClient,
                 retriever: Retriever,
                 catalog_manager: CatalogManager,
                 experiment_store: ExperimentStore):
        self._llm = llm_client
        self._retriever = retriever
        self._catalog = catalog_manager
        self._experiments = experiment_store

    def handle(self, query: str, experiment_id: str) -> Dict:
        """
        完整链路：
        1. FuncCall 检测
        2. LLM 第一轮（判断能否直答 / 输出槽位）
        3. 解析 LLM 输出 → LLMSlots
        4. 直答路径 / Hedging 兜底
        5. 名录查表 + Query 增强
        6. RAG 检索
        7. LLM 第二轮（RAG 增强回答）
        """
        # Step 1: FuncCall
        funcall = detect_funcall(query)
        if funcall:
            return {"type": "funcall", "command": funcall.command,
                    "params": funcall.params}

        # Step 2: LLM 第一轮
        messages = self._experiments.build_messages(experiment_id, query)
        llm_output = self._llm.chat(messages, max_tokens=512)

        # Step 3: 解析
        slots = parse_llm_output(llm_output, query)

        # Step 4: 直答
        if not slots.needs_rag and not slots.hedging:
            return {"type": "answer", "content": slots.answer}

        # Step 5: Hedging 兜底
        if slots.hedging:
            slots.roles = keyword_role_classify(query, top_k=2)
            slots.query = query

        # Step 6: 名录查表 + Query 增强
        search_query = self._catalog.enhance_query(slots)

        # Step 7: RAG 检索
        chunks = self._retriever.search(search_query, slots.roles, top_k=8)
        if not chunks:
            return {"type": "answer",
                    "content": slots.answer or "抱歉，我暂时无法回答这个问题。"}

        # Step 8: LLM 第二轮
        rag_context = "\n\n---\n\n".join(c["content"] for c in chunks)
        messages = self._experiments.build_messages(
            experiment_id, query, rag_context=rag_context)
        final_answer = self._llm.chat(messages, max_tokens=1024)

        return {"type": "answer", "content": final_answer,
                "sources": [c["chunk_id"] for c in chunks]}
```

---

## 四、模块依赖关系

```text
query_handler.py
    ├── funcall_detector.py        (无状态，直接调函数)
    ├── slot_parser.py             (无状态，直接调函数)
    ├── experiment_store.py        (注入实例)
    ├── catalog_manager.py         (注入实例)
    ├── retriever.py               (注入实例)
    │     └── milvus_manager.py    (注入实例，已有)
    └── llm_client.py              (注入实例，已有)

slot_parser.py
    └── chunk_processor.score_role()  (调已有的便捷函数做兜底)
```

所有有状态模块通过构造函数注入，`QueryHandler` 不创建任何依赖。单测时可 mock 任意模块。

---

## 五、函数 vs 类的选择总结

| 模块 | 形式 | 理由 |
|:---|:---|:---|
| `slot_parser.py` | **纯函数** | 无状态，输入字符串输出结构体 |
| `catalog_manager.py` | **类** | 启动时加载 `table.json` 建内存索引 |
| `experiment_store.py` | **类** | 启动时加载 YAML 文件 |
| `retriever.py` | **类** | 持有 MilvusManager 和 embedding client 引用 |
| `query_handler.py` | **类** | 持有所有子模块引用，做编排 |

---

## 六、独立验证方式

每个模块都能独立写 `if __name__ == "__main__"` 测试：

- **`slot_parser.py`**：准备 LLM 输出样本字符串，直接测 `parse_llm_output()`
- **`catalog_manager.py`**：加载 `table.json`，测 `fuzzy_lookup("炭疽杆菌")` 能否命中"炭疽芽孢杆菌"
- **`experiment_store.py`**：加载 YAML，测 `build_system_prompt("pcr")` 输出是否正确
- **`retriever.py`**：连 Milvus，测 `search("BSL-3 培养", roles=["sop"])` 的召回结果
- **`query_handler.py`**：集成测试，mock LLM 返回固定字符串测全链路

---

## 七、完整文件清单

```text
biosafe-rag/
├── experiments/                    ← NEW: 实验知识 YAML 文件目录
│   ├── pcr_amplification.yaml
│   ├── nucleic_acid_electrophoresis.yaml
│   └── ...
│
├── funcall_detector.py             (已完成)
├── slot_parser.py                  ← NEW: LLM 输出解析（纯函数）
├── catalog_manager.py              ← NEW: 名录查表 + Query 增强（类）
├── experiment_store.py             ← NEW: 实验提示词管理（类）
├── retriever.py                    ← NEW: RAG Hybrid 检索（类）
├── query_handler.py                ← NEW: 在线请求主控制器（类，薄编排）
│
├── config.py                       (已有，可能需小幅扩展)
├── llm_client.py                   (已有)
├── milvus_manager.py               (已有)
├── chunk_processor.py              (已有，离线用)
├── split_chunk.py                  (已有，离线用)
├── process_table.py                (已有，离线用)
└── pipeline.py                     (已有，离线编排)
```
