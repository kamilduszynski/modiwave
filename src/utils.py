# Standard Library Imports
import os
from pathlib import Path


def get_repo_path() -> str:
    """Returns absolute path of this repository as a string"""
    current_path = Path(os.path.abspath(".."))
    this_file_path = current_path / Path(__file__)
    return str(this_file_path.parent.parent) + "/"
