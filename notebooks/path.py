"""This script is used in the notebooks to set the path as if the files were in the /src
"""

# Standard Library Imports
import os
import sys

module_path = os.path.abspath(os.pardir)
module_path = os.path.join(module_path, "src")
if module_path not in sys.path:
    sys.path.append(module_path)
