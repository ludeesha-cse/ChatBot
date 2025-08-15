from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Supabase connection
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# VectorStore
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents" 
)

def retrieve_context(query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])
