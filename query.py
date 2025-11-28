"""
===============================================================
  SHAKESPEARE VECTOR DB QUERY SCRIPT
===============================================================

This script allows you to query the pre-built Shakespeare vector database.

Usage:
1. Run the script: python query.py
2. Enter your search queries
3. Type 'exit' or 'quit' to stop

Dependencies:
- chromadb
- sentence-transformers

---------------------------------------------------------------
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

def setup_chroma():
    """Initialize ChromaDB with the existing database."""
    print("Initializing ChromaDB...")

    # Check if database exists
    if not os.path.exists("./shakespeare_chroma_db"):
        print("Error: Shakespeare ChromaDB not found. Please run script.py first to build the database.")
        return None

    # Use PersistentClient for persistent storage
    chroma_client = chromadb.PersistentClient(
        path="./shakespeare_chroma_db"  # where the DB is stored
    )

    # Get the collection
    try:
        collection = chroma_client.get_collection(name="shakespeare")
        print("ChromaDB ready.")
        return collection
    except Exception as e:
        print(f"Error loading collection: {e}")
        return None

def load_model(model_name="all-MiniLM-L6-v2"):
    """Load the sentence transformer model."""
    print(f"Loading embedding model: {model_name} ...")
    try:
        model = SentenceTransformer(model_name)
        print("Model loaded.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def search_shakespeare(query, collection, model, top_k=5):
    """Perform semantic search on the Shakespeare collection."""
    print(f"\n Searching for: '{query}'")
    try:
        q_emb = model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k
        )
        return list(zip(results["distances"][0], results["documents"][0]))
    except Exception as e:
        print(f"Error during search: {e}")
        return []

def main():
    """Main query loop."""
    print("=" * 60)
    print("SHAKESPEARE VECTOR DB QUERY TOOL")
    print("=" * 60)

    # Setup
    collection = setup_chroma()
    if not collection:
        return

    model = load_model()
    if not model:
        return

    print("\nDatabase and model ready! Enter your queries below.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Enter search query: ").strip()
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not query:
                continue

            results = search_shakespeare(query, collection, model, top_k=5)
            if results:
                for i, (dist, passage) in enumerate(results, 1):
                    print(f"\n--- Result {i} (Similarity: {dist:.4f}) ---")
                    # Show first 500 characters to keep output manageable
                    print(f"{passage[:500]}{'...' if len(passage) > 500 else ''}")
                print("\n" + "=" * 60)
            else:
                print("No results found.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
