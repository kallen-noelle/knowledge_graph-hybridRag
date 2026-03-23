from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from build_knowledge_graph import KnowledgeGraphBuilder
from bm25_retriever import get_bm25_retriever
import config

class RAGSystem:
    def __init__(self):
    # 初始化嵌入模型
        self.embeddings = config.embeddings
        # 加载Chroma向量存储
        self.vectorstore = config.vectorstore
        # 初始化语言模型
        self.llm = config.llm
        # 创建检索器
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.top_k_documents})

        self.bm25_retriever = get_bm25_retriever()

        self.prompt = ChatPromptTemplate.from_template("你是一个专业的高血压领域医疗助手，基于以下检索到的信息和资料，"
                    "回答用户的问题。请确保回答准确、专业，并符合最新的医学指南。"
                    "检索到的关系与实体：{graph_context}检索的资料：{text}用户问题："
                    "{question}回答：")
        
        # 构建RAG链
        self.rag_chain = (
             self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def reciprocal_rank_fusion(self, results_list, k=60):
        fused_scores = {}
        
        for results in results_list:
            for rank, doc in enumerate(results):
                doc_id = doc.metadata.get('id', str(doc))
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank)
        
        # 按分数排序
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:10]
    
    def query(self, question):
        # 1. 向量检索
        vector_results = self.retriever.invoke(question)
        # 3. 知识图谱检索
        graph_context = self.knowledge_graph.query_related_entities(question)
        # 2. BM25 关键词检索
        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(question)
        

        # 4. 融合结果（RRF）
        fusion_results = []
        if vector_results or bm25_results:
            fusion_results = self.reciprocal_rank_fusion([vector_results, bm25_results])
        
        # 5. 构建检索结果文本
        # 从融合结果中获取文档内容
        fused_docs = []
        if fusion_results:
            # 创建文档ID到文档的映射
            doc_id_map = {}
            for doc in vector_results + bm25_results:
                doc_id = doc.metadata.get('id', str(doc))
                doc_id_map[doc_id] = doc
            
            # 根据融合结果获取文档
            for doc_id, score in fusion_results:
                if doc_id in doc_id_map:
                    fused_docs.append(doc_id_map[doc_id])
        # 如果融合结果为空，使用向量检索结果
        if not fused_docs:
            fused_docs = vector_results
        
        # 构建文本内容
        documents_text = "\n\n".join([doc.page_content for doc in fused_docs[:2]])
        graph_context = "\n".join([f"  - {entity.page_content}" for entity in graph_context])
        
        # 生成回答
        answer = self.rag_chain.invoke({"question": question, "graph_context": graph_context, "text": documents_text})
        
        # 添加检索信息
        result = f"{answer}\n\n---\n检索到的关系与实体（graph_context）：\n{graph_context}\n\n检索的资料（text）：\n{documents_text}"
        
        return result
    
if __name__ == "__main__":
    # 初始化RAG系统
    rag_system = RAGSystem()
    
    # 测试示例查询
    test_questions = [
        "为一位45岁男性、血压150/95mmHg且无其他合并症的患者推荐初始降压治疗方案。",
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        answer = rag_system.query(question)
        print(f"回答: {answer}")
