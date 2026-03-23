from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

class TextSplitter:
    def __init__(self):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
    
    def split_text(self, text, source):
        """将文本按指定大小分块，返回带有来源信息的分块列表"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            chunks.append({
                'text': chunk,
                'source': source,
                'start': start,
                'end': end
            })
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def split_text_into_chunks(self, text, source):
        """使用 RecursiveCharacterTextSplitter 分块，返回带编号的分块列表"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 每个chunk的最大字符数（可根据模型上下文调整）
            chunk_overlap=100,  # 相邻chunk重叠100字符，避免拆分切断语义
            separators=["\n\n", "\n", ".", "！", "？", ".", "!", "?", ",","，"]  # 优先按段落/句子拆分
        )
        chunks = text_splitter.split_text(text)
        # 给每个chunk加编号，方便后续整合
        chunks=[{
                'text': chunk,
                'source': source,
                'start': 0,
                'end': 0,
            } for chunk in chunks]
        return chunks


if __name__ == "__main__":
    splitter = TextSplitter()
    
    # 测试文本分块
    test_text = "这是一段测试文本，用于测试文本分块功能。" * 50  # 生成一段较长的文本
    source = "test.txt"
    
    chunks = splitter.split_text(test_text, source)
    print(f"原始文本长度: {len(test_text)} 字符")
    print(f"分块数量: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n分块 {i+1}:")
        print(f"  来源: {chunk['source']}")
        print(f"  起始位置: {chunk['start']}")
        print(f"  结束位置: {chunk['end']}")
        print(f"  内容: {chunk['text'][:100]}...")
    
    # 测试 split_text_into_chunks
    print("\n\n测试 split_text_into_chunks:")
    chunks_with_id = splitter.split_text_into_chunks(test_text)
    print(f"分块数量: {len(chunks_with_id)}")

