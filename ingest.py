import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import insert_documents

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DATA_FILE = "data/myfile.txt"

def run_ingestion(supabase):
    # Load document
    loader = TextLoader(DATA_FILE)
    documents = loader.load()

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    # Generate embeddings (1536 dims to match Supabase vector column)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    chunk_vectors = [(chunk.page_content, embeddings.embed_query(chunk.page_content), {"source": DATA_FILE}) for chunk in chunks]

    # Insert into Supabase
    insert_documents(supabase, chunk_vectors)
    print("✅ Ingestion complete.")
