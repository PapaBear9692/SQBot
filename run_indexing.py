import sys
import os

# This ensures that imports like 'from model.dataLoaderModel' work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controller.dataController import run_indexing_pipeline

if __name__ == "__main__":
    run_indexing_pipeline()