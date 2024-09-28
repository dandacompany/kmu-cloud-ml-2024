from typing import List, Dict, Any

# 주어진 파일의 numbers 리스트의 각 숫자에 1을 더한 값으로 suffix 로 붙인 파일이름을 리턴하는 함수
def example(data: Dict[str, Any]) -> List[int]:
    numbers = map(lambda x: x + 1, data.get('numbers', []))
    file_path = data.get('file_path')
    file_full_name = file_path.split('/')[-1]
    file_name = file_full_name.split('.')[0]
    file_extension = file_full_name.split('.')[1]
    return [ file_name + str(number) + '.' + file_extension for number in numbers ]


test_data = {
    'numbers': [1, 2, 3, '4'],
    'file_path': None
}

result = example(test_data)
print(f"함수 실행 결과: {result}")
