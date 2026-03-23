import os
import pdfplumber

class FileProcessor:
    def __init__(self):
        pass
    
    def process_file(self, file_path):
        """处理文件，提取内容和来源信息"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            content = self._process_txt(file_path)
        elif file_extension == '.pdf':
            content = self._process_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")
        
        return {
            'content': content,
            'source': file_path
        }
    
    def _process_txt(self, file_path):
        """处理TXT文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    def _process_pdf(self, file_path):
        """处理PDF文件"""
        content = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    content += page_text
        return content

if __name__ == "__main__":
    processor = FileProcessor()
    
    # # 测试TXT文件处理
    txt_result = processor.process_file('用药实例数据集.txt')
    print(f"TXT文件来源: {txt_result['source']}")
    print(f"TXT内容长度: {len(txt_result['content'])} 字符")
    print(f"TXT内容预览: {txt_result['content'][:500]}...")

    # 测试PDF文件处理
    try:
        pdf_result = processor.process_file('国家基层高血压防治管理指南 2025 版.pdf')
        print(f"\nPDF文件来源: {pdf_result['source']}")
        print(f"PDF内容长度: {len(pdf_result['content'])} 字符")
        print(f"PDF内容预览: {pdf_result['content'][:500]}...")
    except FileNotFoundError:
        print("\nPDF文件未找到，跳过PDF测试")
