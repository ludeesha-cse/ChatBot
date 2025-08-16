from vector_store import init_supabase
from ingest import run_ingestion
from chatbot import run_chatbot
from chatbotWithLogs import run_chatbot_with_logs

def main():
    print("Select an option:")
    print("1) Ingest data")
    print("2) Run chatbot")
    print("3) Run chatbot with logs")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        supabase = init_supabase()
        run_ingestion(supabase)
    elif choice == "2":
        run_chatbot()
    elif choice == "3":
        run_chatbot_with_logs()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
