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
    """
    Insert a list of (text, vector, metadata, text_hash) into Supabase.
    Skips if content_hash already exists.
    """
    from uuid import uuid4
    for text, vector, metadata, text_hash in docs:
        # Check if hash exists in DB
        existing = supabase.table("documents").select("id").eq("content_hash", text_hash).execute()
        if existing.data:
            print(f"Skipped chunk (hash exists): {text_hash}")
            continue

        # Insert new record
        supabase.table("documents").insert({
            "id": str(uuid4()),
            "content": text,
            "embedding": vector,
            "metadata": metadata,
            "content_hash": text_hash,
        }).execute()
        print(f"Inserted chunk (hash: {text_hash})")
