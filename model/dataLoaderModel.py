import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.app_config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIRECTORY

class DataLoader:
    """
    Handles loading all .txt and .pdf files from the data directory
    and splitting them into manageable chunks.
    """
    def __init__(self, data_dir: str = DATA_DIRECTORY):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def load_documents(self) -> List[Document]:
        """Loads all .txt and .pdf files from the specified directory."""
        
        # Ensure the data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Error: Data directory '{self.data_dir}' not found.")
            return []

        print(f"Loading documents from {self.data_dir}...")
        
        # Use DirectoryLoader to handle multiple file types
        # We specify a loader for each file type
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*", # Load all files in all subdirectories
            use_multithreading=True,
            show_progress=True,
            loader = DirectoryLoader(
                self.data_dir,
                loader_cls={
                    ".pdf": PyPDFLoader,
                    ".txt": TextLoader
                }
            )

        )
        
        documents = loader.load()
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of loaded documents into smaller chunks."""
        if not documents:
            return []
            
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        return chunks