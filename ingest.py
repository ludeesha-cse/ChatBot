import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import insert_documents, init_supabase

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DATA_FILE = "data/myfile.txt"

def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def run_ingestion(supabase):

    # Load document
    loader = TextLoader(DATA_FILE)
    documents = loader.load()

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    # Embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",  # 3072-dim output
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Prepare data for insertion
    docs_to_insert = []
    for chunk in chunks:
        text = chunk.page_content
        text_hash = compute_hash(text)
        vector = embeddings.embed_query(text)
        docs_to_insert.append((text, vector, {"source": DATA_FILE}, text_hash))

    # Insert into Supabase (only new content)
    insert_documents(supabase, docs_to_insert)
    print("🚀 Ingestion process finished.")

if __name__ == "__main__":
    run_ingestion()
