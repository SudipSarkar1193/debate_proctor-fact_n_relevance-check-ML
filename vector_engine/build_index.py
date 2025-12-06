import sys
import os

# Add parent directory to path ---
# This allows the script to see 'database' and 'config' in the main folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---------------------------------------------

import time
import pickle
import numpy as np
import faiss
import google.generativeai as genai
import os
from dotenv import load_dotenv
from database import operations as db
from config import DB_CONFIG

# --- CONFIGURATION ---
# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ API_KEY not found in .env file!")

# Model Settings
EMBEDDING_MODEL = "models/text-embedding-004"
VECTOR_DIMENSION = 768  # Standard for Gemini 004 model

# Chunking Settings
CHUNK_SIZE = 800   # Characters per chunk
OVERLAP = 100      # Characters of overlap to keep context
BATCH_LIMIT = 100  # Gemini API limit (items per request)

def setup_gemini():
    """Configures the Gemini API client."""
    genai.configure(api_key=GEMINI_API_KEY)

def simple_chunker(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Splits long text into smaller overlapping chunks.
    Ensures context is preserved between segments.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Stop if we reached the end
        if end >= text_len:
            break
            
        # Move forward, but step back by overlap amount
        start += chunk_size - overlap
        
    return chunks

def get_batch_embeddings(text_chunks):
    """
    Generates embeddings for a list of text chunks using Gemini.
    CRITICAL: Handles the 100-item API limit by sub-batching.
    """
    if not text_chunks:
        return []

    all_vectors = []
    
    # Process in safe sub-batches (e.g., 0-100, 100-200, etc.)
    for i in range(0, len(text_chunks), BATCH_LIMIT):
        batch = text_chunks[i : i + BATCH_LIMIT]
        
        try:
            # Gemini API call
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )

            print(f"🧩 Processed sub-batch {i // BATCH_LIMIT + 1} ({len(batch)} items)")
            
            # Extract vectors and add to master list
            if 'embedding' in result:
                print(f"🧩 Result Structure: {result.keys()}") 
                # Show first vector's first 5 dimensions just to see what it looks like
                print(f"👀 Sample Vector (First 5 dims): {result['embedding'][0][:5]} ...")

                print()
                print("----"*15)
                print()
                if len(result['embedding']) > 1:
                    print(f"👀 Sample Vector (First 5 dims): {result['embedding'][1][:5]} ...")
                all_vectors.extend(result['embedding'])
            
            # Polite sleep between sub-batches
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ⚠️ API Error on sub-batch {i}: {e}")
            # We continue to the next batch instead of crashing
            continue

    return all_vectors

def build_index():
    print("🚀 Starting Phase 2: Building Vector Index...")
    setup_gemini()
    
    # 1. Fetch Data
    print("📥 Fetching data from PostgreSQL...")
    conn = db.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT pageid, title, content, url FROM raw_facts")
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    print(f"   Found {len(rows)} articles. Beginning vectorization...")

    # FAISS Data Structures
    master_embedding_list = []
    metadata_list = [] 
    
    global_faiss_id = 0
    
    # 2. Process Articles
    for index, row in enumerate(rows):
        page_id, title, content, url = row
        
        # A. Chunking
        chunks = simple_chunker(content)
        if not chunks:
            continue

        # B. Embedding (Robust Batching)
        vectors = get_batch_embeddings(chunks)
        
        # Validation: vectors count must match chunks count (roughly)
        # If API failed for some batches, we truncate chunks to match vectors length
        if len(vectors) != len(chunks):
            # Safe fallback: only use chunks we successfully vectorized
            chunks = chunks[:len(vectors)]

        if not vectors:
            continue
            
        # C. Store in Memory
        for i, vector in enumerate(vectors):
            master_embedding_list.append(vector)
            
            # Map this vector ID to the actual text info
            metadata_list.append({
                "faiss_id": global_faiss_id,
                "title": title,
                "text": chunks[i],
                "url": url
            })
            global_faiss_id += 1
            
        print(f"   [{index+1}/{len(rows)}] Processed: {title} ({len(chunks)} chunks)")
        
        # Rate Limit Protection (Sleep between articles)
        time.sleep(1.0) 

    # 3. Build & Save FAISS Index
    total_vectors = len(master_embedding_list)
    if total_vectors == 0:
        print("❌ No embeddings were generated. Check API key or Database.")
        return

    print(f"\n🧠 Building FAISS Index with {total_vectors} vectors...")
    
    # Convert to float32 numpy array (Required by FAISS)
    embedding_matrix = np.array(master_embedding_list).astype('float32')
    
    # Create Index
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embedding_matrix)
    
    # 4. Save to Disk
    print("💾 Saving files to disk...")
    faiss.write_index(index, "vector_store.index")
    
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata_list, f)
        
    print(f"✅ DONE! Saved 'vector_store.index' and 'metadata.pkl' ({total_vectors} chunks).")

if __name__ == "__main__":
    build_index()