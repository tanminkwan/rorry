"""
아래 조건을 만족하는 Reader class 를 만들어라
---
1. 생성자
- pdf file 명을 argument로 받는다
- langchain_community.document_loaders.PyPDFLoader로 loader 생성.
2. method get_iterator
- chunk_size, chunk_overlap를 argument로 받는다.
- pdf file을 load하고 전체 텍스트를 하나의 문자열로 결합
- chunk_size, chunk_overlap를 적용한 text splitter 실행
- chunk를 하나씩 리턴하는 iterator return 
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Reader:
    def __init__(self, pdf_path):
        # PDF 파일 경로를 받아 PyPDFLoader 초기화
        self.loader = PyPDFLoader(pdf_path)
    
    def get_iterator(self, chunk_size=1000, chunk_overlap=100):
        # PDF 파일을 로드하여 전체 텍스트 추출
        pages = self.loader.load()

        # 전체 텍스트를 하나의 문자열로 결합
        full_text = ''.join(page.page_content for page in pages)

        # 텍스트 스플리터 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 전체 텍스트를 나누기
        chunks = text_splitter.split_text(full_text)

        # 각 덩어리를 yield
        for chunk in chunks:
            yield chunk

if __name__ == "__main__":
    
    reader = Reader('2021_korea_bank.pdf').get_iterator(chunk_size=5000, chunk_overlap=500)
    for i, row_data in enumerate(reader):
        print("\n\n\n[chunk] #",i)
        print(row_data)
