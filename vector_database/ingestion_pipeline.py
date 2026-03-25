import json
import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "FAQs", "resolution_faqs.json")
JSON_FILE = file_path
DB_PATH = "./chroma_db"
COLLECTION_NAME = "sre_runbooks"
# all-MiniLM-L6-v2 is an excellent, fast HuggingFace model for local semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

def build_vector_db():
    print(f"Loading FAQs from {JSON_FILE}...")
    
    if not os.path.exists(JSON_FILE):
        raise FileNotFoundError(f"Could not find {JSON_FILE}. Please run the FAQ generator script first.")

    with open(JSON_FILE, "r") as f:
        faqs = json.load(f)

    # 2. Prepare the data for ChromaDB
    documents =[]
    metadatas = []
    ids =[]

    for faq in faqs:
        # Concatenate into a rich text block so the embedding captures full context
        content = f"Category: {faq['category']}\nQuestion: {faq['question']}\nResolution: {faq['answer']}"
        documents.append(content)
        
        # Store metadata for hybrid filtering (e.g., if the agent only wants to search "Database" issues)
        metadatas.append({
            "category": faq["category"],
            "faq_id": faq["faq_id"]
        })
        
        ids.append(faq["faq_id"])

    print(f"Prepared {len(documents)} documents for indexing.")

    # 3. Initialize ChromaDB Persistent Client
    print(f"Initializing ChromaDB persistent client at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # 4. Define the HuggingFace Embedding Function
    print(f"Downloading/Loading HuggingFace embedding model: {EMBEDDING_MODEL}...")
    hf_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # 5. Create (or get) the collection
    # We use get_or_create so you can run this script multiple times safely
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=hf_embedding_function,
        metadata={"hnsw:space": "cosine"} # Cosine similarity is best for text embeddings
    )

    # 6. Upsert data into the database
    print("Embedding and indexing documents into ChromaDB. This may take a few seconds...")
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ Successfully built and persisted the ChromaDB vector index!\n")

    # ==========================================
    # TEST THE RAG SEARCH FOR THE AI AGENT
    # ==========================================
    print("--- Testing the Vector Search ---")
    
    # Simulating what the Resolver Agent will ask based on the logs we generated earlier
    agent_query = "We had a deployment and now processPayment is failing with high payment gateway latency and PaymentGatewayTimeoutException."
    print(f"Agent Query: '{agent_query}'\n")

    results = collection.query(
        query_texts=[agent_query],
        n_results=1 # We just want the absolute best match
    )

    # Print the results
    if results['documents'][0]:
        best_match = results['documents'][0][0]
        match_id = results['ids'][0][0]
        distance = results['distances'][0][0]
        
        print(f"🔍 Top Result Found ({match_id}) - Distance score: {distance:.4f}")
        print("-" * 40)
        print(best_match)
        print("-" * 40)
    else:
        print("No results found.")

if __name__ == "__main__":
    build_vector_db()