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

class Reader:
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

if __name__ == "__main__":
    excel_file = "qaDf.xlsx"

    with Reader(excel_file) as reader:
        for row_data in reader.get_iterator("구분", "질문", "답"):
            print(row_data)
