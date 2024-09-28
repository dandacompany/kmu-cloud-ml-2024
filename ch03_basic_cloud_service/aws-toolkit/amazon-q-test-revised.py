import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Generator

def traverse_directory(parent_dir: Path) -> Generator[Path, None, None]:
    """
    Traverse the directory tree starting from the given parent directory.

    Yields:
        Path: The next file or directory in the directory tree.
    """
    with os.scandir(parent_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                yield from traverse_directory(entry.path)
            else:
                yield entry.path

def print_directory_tree(parent_dir: Path) -> None:
    """
    Prints the folder and file structure of the given parent directory.
    """
    if not parent_dir.exists() or not parent_dir.is_dir():
        print(f"Error: '{parent_dir}' is not a valid directory.")
        return

    for path in traverse_directory(parent_dir):
        path_obj = Path(path)  # Convert string to Path object
        indent = ' ' * 4 * len(path_obj.relative_to(parent_dir).parts)
        print(f"{indent}{path_obj.name}{' (directory)' if path_obj.is_dir() else ''}")

if __name__ == '__main__':
    parent_dir = Path("./")
    print_directory_tree(parent_dir)
