# 高血压知识图谱与RAG系统

## 项目概述

本项目使用LangChain框架实现了一个基于知识图谱和RAG（检索增强生成）的高血压领域医疗助手。系统可以从多个文件中提取医学实体和关系，构建知识图谱，并使用RAG技术回答用户的医学问题。

## 功能特点

1. **多文件支持**：系统不局限于特定文件，可以处理多种类型的文件（当前支持TXT文件）
2. **大模型实体和关系提取**：使用阿里千问API从文本中提取医学实体和关系
3. **文件来源追踪**：在存储数据时添加来源文件信息，方便溯源
4. **文本分块存储**：将文本按256字划分，重叠64字，存储到ChromaDB
5. **知识图谱检索**：通过知识图谱检索相关信息
6. **RAG增强**：在提示词中同时包含知识图谱检索信息和原始检索资料

## 项目结构

```
├── build_knowledge_graph.py  # 知识图谱构建模块
├── rag_system.py            # RAG系统模块
├── file_processor.py        # 文件处理模块
├── text_splitter.py         # 文本分块模块
├── test_system.py           # 测试模块
├── parse_dataset.py         # 数据集解析模块（保留用于兼容性）
└── README.md                # 项目说明文档
```

## 核心模块说明

### 1. 文件处理模块 (file_processor.py)

负责处理不同类型的文件，提取内容和来源信息。当前支持TXT文件，可扩展支持其他文件类型。

### 2. 文本分块模块 (text_splitter.py)

将文本按256字划分，重叠64字，生成带有来源信息的分块列表。

### 3. 知识图谱构建模块 (build_knowledge_graph.py)

- 使用大模型从文本中提取实体和关系
- 构建知识图谱
- 将分块存储到ChromaDB
- 提供知识图谱查询功能

### 4. RAG系统模块 (rag_system.py)

- 从ChromaDB检索相关信息
- 构建包含知识图谱检索信息和原始检索资料的提示词
- 使用大模型生成回答

## 环境要求

- Python 3.8+
- 依赖库：
  - langchain
  - langchain_chroma
  - langchain_community
  - networkx
  - dashscope

## 安装步骤

1. 克隆项目到本地
2. 安装依赖：
   ```bash
   pip install langchain langchain_chroma langchain_community networkx dashscope
   ```
3. 配置API密钥：
   - 在 `build_knowledge_graph.py` 和 `rag_system.py` 中设置 `dashscope_api_key`

## 使用方法

### 1. 构建知识图谱

```python
from build_knowledge_graph import KnowledgeGraphBuilder

# 初始化构建器
builder = KnowledgeGraphBuilder()

# 从文件构建知识图谱
file_paths = ['文件1.txt', '文件2.txt']  # 可以添加多个文件
graph = builder.build_from_files(file_paths)

# 打印知识图谱统计信息
stats = builder.get_graph_statistics()
print(f"实体数量: {stats['num_nodes']}")
print(f"关系数量: {stats['num_edges']}")

# 将知识图谱持久化到ChromaDB
builder.graph_persist_to_chroma()
```

### 2. 使用RAG系统查询

```python
from rag_system import RAGSystem

# 初始化RAG系统
rag_system = RAGSystem()

# 测试查询
questions = [
    "为一位45岁男性、血压150/95mmHg且无其他合并症的患者推荐初始降压治疗方案。",
    "高血压合并糖尿病患者应如何选择降压药物？",
    "老年高血压患者的治疗注意事项有哪些？"
]

for question in questions:
    answer = rag_system.query(question)
    print(f"问题: {question}")
    print(f"回答: {answer}")
    print("-" * 50)
```

## 注意事项

1. **API密钥**：需要配置有效的阿里千问API密钥
2. **文件类型**：当前仅支持TXT文件，其他文件类型需要扩展文件处理模块
3. **性能**：处理大量文件时可能需要较长时间，建议分批处理
4. **存储**：ChromaDB会在项目目录下创建 `chroma_db` 文件夹存储数据

## 测试

运行测试脚本验证系统功能：

```bash
python test_system.py
```

## 扩展建议

1. 支持更多文件类型（如PDF、DOCX等）
2. 优化大模型实体和关系提取的准确性
3. 添加更多医学领域的实体类型和关系类型
4. 实现更高效的知识图谱查询算法
5. 增加用户界面，方便非技术用户使用

## 许可证

本项目仅供学习和研究使用。