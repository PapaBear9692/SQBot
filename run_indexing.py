import sys
import os

# This ensures that imports like 'from model.dataLoaderModel' work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controller.dataController import run_indexing_pipeline

if __name__ == "__main__":
    """
    This is the main entry point to run the data indexing pipeline.
    Execute this file once from your terminal to populate Pinecone.
    
    Example:
    (SQBot) $ python run_indexing.py
    """
    run_indexing_pipeline()