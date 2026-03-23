import networkx as nx
import json
import uuid
from file_processor import FileProcessor
from text_splitter import TextSplitter

import config
class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        # 使用DashScope Embeddings
        self.embeddings = config.embeddings
        # 初始化Chroma向量存储
        self.vectorstore = config.vectorstore
        self.graphstore = config.graphstore
        # 初始化文件处理器和文本分块器
        self.file_processor = FileProcessor()
        self.text_splitter = TextSplitter()
        # 初始化语言模型
        self.llm = config.llm
    
    def build_from_files(self, file_paths):
        """从多个文件构建知识图谱"""
        for file_path in file_paths:
            # 处理文件
            file_data = self.file_processor.process_file(file_path)
            content = file_data['content']
            source = file_data['source']
            
            # 文本分块
            chunks = self.text_splitter.split_text_into_chunks(content, source)
            
            # 处理每个分块
            for chunk in chunks:
                # 使用大模型提取实体和关系
                entities, relations = self.extract_entities_relations(chunk['text'])

                # 添加实体
                for entity in entities:
                    if isinstance(entity, dict) and "name" in entity and "type" in entity:
                         self.graph.add_node(entity['name'], type=entity['type'], source=f"{chunk['source']}:{chunk['start']}-{chunk['end']}")


                # 添加关系
                for relation in relations:
                    # 检查关系格式是否正确
                    if isinstance(relation, dict) and 'source' in relation and 'target' in relation and 'type' in relation:
                        source_entity = relation['source']
                        target_entity = relation['target']
                        rel_type = relation['type']

                        # 确保源节点和目标节点都有type属性
                        if not self.graph.has_node(source_entity):
                            self.graph.add_node(source_entity, type="unknown", source=source)
                        if not self.graph.has_node(target_entity):
                            self.graph.add_node(target_entity, type="unknown", source=source)

                        self.graph.add_edge(source_entity, target_entity, type=rel_type, source=f"{chunk['source']}:{chunk['start']}-{chunk['end']}")
                
                # 将分块存储到Chroma
                print(chunk)
                self.store_chunk(chunk)
        return self.graph
    
    def extract_entities_relations(self, text):
        """使用大模型提取实体和关系"""
        prompt = (f"从以下文本中提取医学实体和关系，实体类型包括药物、疾病、指南、患者。关系类型包括INDICATED_FOR（药物-适应症）"
                  f"、TREATED_WITH（疾病-治疗方案）、RECOMMENDS（指南-推荐）等。返回格式为JSON，包含entities和relations两个字段。"
                  f"例如：{{entities: [{{name: '药物A', type: '药物'}}, {{name: '疾病B', type: '疾病'}}, {{name: '指南C', type: '指南'}}],"
                  f" relations: [{{source: '药物A', target: '疾病B', type: 'INDICATED_FOR'}}, {{source: '疾病B', target: '指南C', type: 'RECOMMENDS'}}]}}"
                  f"请严格保证按照给定格式返回，不能包含其他内容。"
                  f"\n\n文本：{text}")
        
        response = self.llm.invoke(prompt)
        print(text, response.content)
        # 解析响应
        try:
            result = json.loads(response.content)
            

            return result.get('entities', []), result.get('relations', [])
        except json.JSONDecodeError:
            # 如果解析失败，返回空结果
            return [], []
    """将分块存储到Chroma"""
    def store_chunk(self, chunk):
        """将分块存储到Chroma"""
        text = chunk['text']
        source = chunk['source']
        start = chunk['start']
        end = chunk['end']
        
        # 生成唯一ID
        chunk_id = str(uuid.uuid4())
        
        # 向ChromaDB添加数据
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{"type": "chunk", "source": source, "start": start, "end": end}],
            ids=[chunk_id]
        )
    """将知识图谱持久化到ChromaDB"""
    def graph_persist_to_chroma(self):
        # 准备数据
        texts = []
        metadatas = []
        ids = []
        
        # 添加实体
        for node, data in self.graph.nodes(data=True):
            text = f"实体: {node}, 类型: {data.get('type', 'unknown')}"
            texts.append(text)
            metadatas.append({"type": "entity", "entity_type": data.get('type', 'unknown'), "source": data.get('source', 'unknown')})
            ids.append(f"entity_{node}")
        
        # 添加关系
        for u, v, data in self.graph.edges(data=True):
            text = f"关系: {u} -> {v}, 类型: {data.get('type', 'unknown')}"
            texts.append(text)
            metadatas.append({"type": "relation", "relation_type": data.get('type', 'unknown'), "source": data.get('source', 'unknown')})
            ids.append(f"relation_{u}_{v}")
        
        # 向ChromaDB添加数据
        self.graphstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"知识图谱已持久化到ChromaDB，包含 {self.graph.number_of_nodes()} 个实体和 {self.graph.number_of_edges()} 个关系")
    
    def query_related_entities(self, entity, top_k=config.top_k_graph):
        """查询与指定实体相关的实体"""
        # 从ChromaDB查询
        results = self.graphstore.similarity_search(
            query=f"与 {entity} 相关的实体",
            k=top_k
        )
        
        return results
    
    def get_graph_statistics(self):
        """获取知识图谱统计信息"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": {type: len([n for n, d in self.graph.nodes(data=True) if d.get('type') == type])
                           for type in set(d.get('type') for n, d in self.graph.nodes(data=True))}
        }

if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    
    # 从文件构建知识图谱
    file_paths = ["国家基层高血压防治管理指南 2025 版.pdf"]
    graph = builder.build_from_files(file_paths)
    


