from langchain_openai import OpenAIEmbeddings
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

def retrieve_context(query, k=3):
    """Retrieve context using direct Supabase RPC call instead of LangChain wrapper."""
    try:
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(query)
        
        # Call the Supabase RPC function with correct parameter order
        result = supabase.rpc(
            "match_documents",
            {
                "match_count": k,
                "query_embedding": query_embedding
            }
        ).execute()
        
        if result.data:
            # Extract content from the results
            documents = []
            for doc in result.data:
                if 'content' in doc:
                    documents.append(doc['content'])
                elif 'page_content' in doc:  # Fallback
                    documents.append(doc['page_content'])
            
            return "\n\n".join(documents)
        else:
            return "No relevant documents found."
            
    except Exception as e:
        print(f"Error in retrieve_context: {e}")
        return f"Retrieval error: {e}"
