from rank_bm25 import BM25Okapi
import jieba
import config
class BM25Retriever:
    def __init__(self, documents):
        """
        初始化 BM25 检索器
        documents: 文档列表，每个文档包含 page_content 和 metadata
        """
        self.documents = documents
        # 对文档进行分词
        self.tokenized_docs = [
            list(jieba.cut(doc.page_content)) 
            for doc in documents
        ]
        # 创建 BM25 模型
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def retrieve(self, query, top_k=config.top_k_bm25):
        """
        根据查询检索相关文档
        query: 查询字符串
        top_k: 返回前k个结果
        """
        # 对查询进行分词
        tokenized_query = list(jieba.cut(query))
        
        # 获取得分
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top-k结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # 返回结果
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有得分的文档
                results.append(self.documents[idx])
        
        return results
def get_bm25_retriever():
        bm25_retriever = None
        all_docs = config.vectorstore._collection.get()
        if 'documents' in all_docs and 'metadatas' in all_docs:
            from langchain_core.documents import Document
            documents = []
            for content, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            bm25_retriever = BM25Retriever(documents)
        return bm25_retriever
    