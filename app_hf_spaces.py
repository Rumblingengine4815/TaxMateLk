# HuggingFace Spaces entry point
# Spaces looks for app.py in the root directory
import os

import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.app import launch

if __name__ == "__main__":
    launch()
