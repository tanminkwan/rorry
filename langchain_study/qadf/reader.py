"""
아래 조건을 만족하는 Reader class 를 만들어라
---
1. 생성자
- excel file 명을 argument로 받는다
- excel file 을 오픈하고 file 핸들을 private 값에 저장한다.
2. method get_iterator
- 조회할 column 명들을 argument로 받는다. (개수 무제한)
- 각각의 row를 해당 {column명 : cell값} dictionary에 담아 yield 하는 iterator를 반환한다. 
"""
import openpyxl

class ExcelReader:
    def __init__(self, file_name):
        self._file_name = file_name
        self._excel_data = None

    def __enter__(self):
        self._excel_data = openpyxl.load_workbook(self._file_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._excel_data:
            self._excel_data.close()

    def get_iterator(self, *column_names):
        if not self._excel_data:
            raise Exception("Excel file is not open.")

        sheet = self._excel_data.active
        headers = [cell.value for cell in sheet[1]]  # 첫 번째 행의 칼럼명들을 가져옵니다

        column_indices = []
        for column_name in column_names:
            try:
                column_index = headers.index(column_name) + 1
                column_indices.append(column_index)
            except ValueError:
                raise ValueError(f"Column '{column_name}' not found in the sheet.")

        for row in sheet.iter_rows(min_row=2, values_only=True):  # 첫 번째 행은 헤더이므로 2번째 행부터 데이터입니다
            row_data = {}
            for column_name, column_index in zip(column_names, column_indices):
                row_data[column_name] = row[column_index - 1]  # column_index는 1부터 시작하므로 -1 해줍니다
            yield row_data

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

class PDFReader:
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
    excel_file = "qaDf.xlsx"

    with Reader(excel_file) as reader:
        for row_data in reader.get_iterator("구분", "질문", "답"):
            print(row_data)
