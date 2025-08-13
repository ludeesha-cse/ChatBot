import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def init_supabase():
    """Initialize Supabase client"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_documents(supabase, docs):
    """Insert a list of (text, vector, metadata) into Supabase"""
    from uuid import uuid4
    for text, vector, metadata in docs:
        supabase.table("documents").insert({
            "id": str(uuid4()),
            "content": text,
            "embedding": vector,
            "metadata": metadata
        }).execute()
