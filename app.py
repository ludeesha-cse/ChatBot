from vector_store import init_supabase
from ingest import run_ingestion

def main():
    supabase = init_supabase()
    run_ingestion(supabase)

if __name__ == "__main__":
    main()
