import os
import warnings

from src.app import launch

if __name__ == "__main__":
    print("Launching TaxMate LK UI...")
    
    
    # Suppress the annoying internal Gradio/Starlette warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    launch()
