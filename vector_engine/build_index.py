import sys
import os
import argparse 

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv

import time
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from database import operations as db
from config import TOPIC_REGISTRY 

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ API_KEY not found in .env file!")

# Model Settings
EMBEDDING_MODEL = "models/text-embedding-004"
VECTOR_DIMENSION = 768

# Chunking Settings
CHUNK_SIZE = 800
OVERLAP = 100
BATCH_LIMIT = 100

def setup_gemini():
    genai.configure(api_key=GEMINI_API_KEY)

def simple_chunker(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    if not text: return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len: break
        start += chunk_size - overlap
    return chunks

def get_batch_embeddings(text_chunks):
    if not text_chunks: return []
    all_vectors = []
    for i in range(0, len(text_chunks), BATCH_LIMIT):
        batch = text_chunks[i : i + BATCH_LIMIT]
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )
            if 'embedding' in result:
                all_vectors.extend(result['embedding'])
            time.sleep(0.5)
        except Exception as e:
            print(f"   ⚠️ API Error on sub-batch {i}: {e}")
            continue
    return all_vectors

def build_index():
    # 1. Parse Args
    parser = argparse.ArgumentParser(description="Build Vector Index for a specific topic.")
    parser.add_argument("--topic", type=str, required=True, help="The topic key (e.g., 'ai', 'aadhaar')")
    args = parser.parse_args()
    topic = args.topic.lower()

    if topic not in TOPIC_REGISTRY:
        print(f"❌ Error: Topic '{topic}' not found in config.")
        return

    print(f"🚀 Starting Phase 2: Building Brain for [{topic.upper()}]...")
    setup_gemini()
    
    # Define Paths
    output_dir = os.path.join("data", topic)
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "vector_store.index")
    meta_path = os.path.join(output_dir, "metadata.pkl")

    # --- 2. SMART LOADING (INCREMENTAL LOGIC) ---
    master_embedding_list = []
    metadata_list = []
    existing_titles = set()
    
    # Try to load existing brain to append to it
    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("🧠 Found existing brain. Loading for incremental update...")
        try:
            # Load FAISS Index
            index = faiss.read_index(index_path)
            
            # Load Metadata
            with open(meta_path, "rb") as f:
                metadata_list = pickle.load(f)
            
            # Create a set of titles we already have
            existing_titles = {item['title'] for item in metadata_list}
            print(f"   ✅ Loaded {len(metadata_list)} existing memories ({len(existing_titles)} unique articles).")
            
        except Exception as e:
            print(f"   ⚠️ Error loading existing index: {e}. Starting fresh.")
            index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    else:
        print("🆕 No existing brain found. Creating fresh index.")
        index = faiss.IndexFlatL2(VECTOR_DIMENSION)

    # 3. Fetch Data (from specific DB)
    print(f"📥 Fetching data from {TOPIC_REGISTRY[topic]['db_config']['dbname']}...")
    conn = db.get_connection(topic)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT pageid, title, content, url FROM raw_facts")
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    # 4. Filter: Only process NEW articles
    new_rows = [r for r in rows if r[1] not in existing_titles]
    
    if not new_rows:
        print("✨ Brain is already up to date! No new articles to embed.")
        return

    print(f"   Found {len(rows)} total articles. {len(new_rows)} are NEW. Processing...")

    # Calculate starting ID for FAISS (continuation)
    global_faiss_id = len(metadata_list)
    new_vectors_count = 0
    
    # 5. Process ONLY New Articles
    for idx, row in enumerate(new_rows):
        page_id, title, content, url = row
        
        chunks = simple_chunker(content)
        if not chunks: continue

        vectors = get_batch_embeddings(chunks)
        
        if len(vectors) != len(chunks):
            chunks = chunks[:len(vectors)]

        if not vectors: continue
            
        for i, vector in enumerate(vectors):
            master_embedding_list.append(vector)
            metadata_list.append({
                "faiss_id": global_faiss_id,
                "title": title,
                "text": chunks[i],
                "url": url
            })
            global_faiss_id += 1
            new_vectors_count += 1
            
        print(f"   [{idx+1}/{len(new_rows)}] Processed: {title} ({len(chunks)} chunks)")
        time.sleep(1.0) 

    # 6. Save Updates
    if new_vectors_count > 0:
        print(f"\n🧠 Adding {new_vectors_count} new vectors to existing index...")
        
        # Add new vectors to the existing FAISS index
        new_matrix = np.array(master_embedding_list).astype('float32')
        index.add(new_matrix)
        
        print(f"💾 Saving updated brain to: {output_dir}")
        faiss.write_index(index, index_path)
        
        with open(meta_path, "wb") as f:
            pickle.dump(metadata_list, f)
            
        print(f"✅ DONE! Brain updated. Total memories: {index.ntotal}")
    else:
        print("⚠️ No valid vectors generated from new content.")

if __name__ == "__main__":
    build_index()