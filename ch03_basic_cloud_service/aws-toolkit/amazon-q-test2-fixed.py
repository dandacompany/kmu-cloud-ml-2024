from typing import List, Dict, Any

# Function to append a suffix to a file name based on a list of numbers
def example(data: Dict[str, Any]) -> List[str]:
    numbers = [int(num) + 1 for num in data.get('numbers', [])]  # Convert elements to int and add 1
    file_path = data.get('file_path')

    if file_path is None:
        return []  # Return an empty list if file_path is None

    file_full_name = file_path.split('/')[-1]
    file_name, file_extension = file_full_name.split('.')

    return [f"{file_name}{number}.{file_extension}" for number in numbers]


test_data = {
    'numbers': [1, 2, 3, '4'],
    'file_path': '/path/to/file.ext'
}

result = example(test_data)
print(f"Function result: {result}")
