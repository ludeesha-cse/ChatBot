from vector_store import init_supabase
from ingest import run_ingestion
from chatbot import run_chatbot

def main():
    print("Select an option:")
    print("1) Ingest data")
    print("2) Run chatbot")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        supabase = init_supabase()
        run_ingestion(supabase)
    elif choice == "2":
        run_chatbot()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
