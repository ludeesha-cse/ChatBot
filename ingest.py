import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import compute_hash, load_documents_from_folder
from vector_store import insert_documents, init_supabase

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DATA_FOLDER = "./data/"

def run_ingestion(supabase):

    # Load document
    documents = load_documents_from_folder(DATA_FOLDER)

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    # Embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Prepare data for insertion
    docs_to_insert = []
    for chunk in chunks:
        text = chunk.page_content
        text_hash = compute_hash(text)
        vector = embeddings.embed_query(text)
        docs_to_insert.append((text, vector, {"source": chunk.metadata.get("source")}, text_hash))

    # Insert into Supabase (only new content)
    insert_documents(supabase, docs_to_insert)
    print("Ingestion process finished.")

if __name__ == "__main__":
    supabase = init_supabase()
    run_ingestion(supabase)
