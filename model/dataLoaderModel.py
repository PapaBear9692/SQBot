from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from core.interfaces import DataLoaderInterface, TextSplitterInterface
import config

class SimpleDirectoryLoader(DataLoaderInterface):
    """Loads all .txt files from a directory."""
    def load(self, source: str) -> List[Document]:
        print(f"Loading documents from: {source}")
        loader = DirectoryLoader(
            source, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True
        )
        try:
            return loader.load()
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []

class LangChainTextSplitter(TextSplitterInterface):
    """Wraps LangChain's RecursiveCharacterTextSplitter."""
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        print(f"Text splitter initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        print(f"Splitting {len(documents)} documents...")
        return self.splitter.split_documents(documents)

# --- Factory Functions ---

def get_loader() -> DataLoaderInterface:
    return SimpleDirectoryLoader()

def get_splitter() -> TextSplitterInterface:
    return LangChainTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
