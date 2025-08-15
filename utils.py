import os
from langchain_unstructured import UnstructuredLoader

import hashlib

def load_documents_from_folder(folder_path: str):
    """Load all supported documents from a folder."""
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            loader = UnstructuredLoader(file_path)
            documents.extend(loader.load())
    
    return documents


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()