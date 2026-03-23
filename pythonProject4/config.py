from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi

embeddings = DashScopeEmbeddings(dashscope_api_key="", model="text-embedding-v4")
llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=""
)
vectorstore = Chroma(
    collection_name="vectorstoreV2",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
graphstore = Chroma(
            collection_name="graphstore",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
)
chunk_size = 256
chunk_overlap = 64
top_k_graph = 5
top_k_documents = 3
top_k_bm25 = 3
