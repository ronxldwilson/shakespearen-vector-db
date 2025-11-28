"""
===============================================================
  SHAKESPEARE SEMANTIC SEARCH WITH CHROMADB (LOCAL EMBEDDINGS)
===============================================================

This script:
1. Downloads Shakespeare text from Gutenberg
2. Chunks it (~400 tokens per chunk with 40-token overlap)
3. Generates embeddings using sentence-transformers
4. Stores embeddings and text in ChromaDB (DuckDB + Parquet)
5. Allows semantic search with a function
6. Shows progress while generating embeddings
---------------------------------------------------------------
"""

import requests
from tqdm import tqdm
import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

# -----------------------------------------------------
# 1. DOWNLOAD SHAKESPEARE
# -----------------------------------------------------
def download_shakespeare():
    print("Downloading Shakespeare corpus...")
    url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    text = requests.get(url).text
    print(" Download complete.")
    return text

# -----------------------------------------------------
# 2. CHUNK TEXT
# -----------------------------------------------------
def chunk_text(text, max_tokens=400, overlap=40):
    print("Chunking text...")
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    current = []
    current_tokens = 0

    for word in words:
        token_len = len(enc.encode(word))
        if current_tokens + token_len > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap:]
            current_tokens = sum(len(enc.encode(w)) for w in current)
        current.append(word)
        current_tokens += token_len

    if current:
        chunks.append(" ".join(current))

    print(f" Chunking complete — total chunks: {len(chunks)}")
    return chunks

# -----------------------------------------------------
# 3. SETUP CHROMADB
# -----------------------------------------------------
def setup_chroma():
    print(" Initializing ChromaDB...")

    # Use PersistentClient for persistent storage (Chroma v0.4+)
    chroma_client = chromadb.PersistentClient(
        path="./shakespeare_chroma_db"  # where the DB will be stored
    )

    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name="shakespeare",
        metadata={"hnsw:space": "cosine"}
    )

    print(" ChromaDB ready.")
    return collection
# -----------------------------------------------------
# 4. GENERATE EMBEDDINGS WITH PROGRESS
# -----------------------------------------------------
def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2", batch_size=50):
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    embeddings = []
    print("Generating embeddings...")
    for i in tqdm(range(0, len(chunks), batch_size), ncols=80, desc="Embedding batches"):
        batch = chunks[i:i+batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())
    print(" Embeddings generated.")
    return embeddings

# -----------------------------------------------------
# 5. INSERT INTO CHROMADB
# -----------------------------------------------------
def insert_into_chroma(chunks, embeddings, collection, batch_size=50):
    print("Inserting chunks into ChromaDB...")
    for i in tqdm(range(0, len(chunks), batch_size), ncols=80, desc="Inserting batches"):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
        collection.add(
            ids=ids,
            documents=batch_chunks,
            embeddings=batch_embeddings
        )
    print(" All chunks inserted.")

# -----------------------------------------------------
# 6. SEMANTIC SEARCH FUNCTION
# -----------------------------------------------------
def search_shakespeare(query, collection, model, top_k=5):
    print(f"\n🔍 Searching for: '{query}'")
    q_emb = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    return list(zip(results["distances"][0], results["documents"][0]))

# -----------------------------------------------------
# 7. MAIN PIPELINE
# -----------------------------------------------------
if __name__ == "__main__":
    # Step 1
    text = download_shakespeare()

    # Step 2
    chunks = chunk_text(text)

    # Step 3
    collection = setup_chroma()

    # Step 4
    embeddings = generate_embeddings(chunks)

    # Step 5
    insert_into_chroma(chunks, embeddings, collection)

    # Step 6 — Test searches
    model = SentenceTransformer("all-MiniLM-L6-v2")  # reuse for search
    queries = [
        "passages about ambition",
        "themes of betrayal",
        "love and romance",
        "existential fear"
    ]

    for q in queries:
        results = search_shakespeare(q, collection, model, top_k=3)
        for dist, passage in results:
            print(f"\n--- Similarity: {dist:.4f} ---\n{passage[:400]}...\n")
        print("\n" + "="*60 + "\n")
