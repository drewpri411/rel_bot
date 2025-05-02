import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import uuid
import time

# ---- Step 1: Load data ----
with open("data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# ---- Step 2: Load embedding model ----
model = SentenceTransformer('all-MiniLM-L6-v2')  # Output dim = 384

# ---- Step 3: Init Pinecone ----
pc = Pinecone(api_key="YOUR_API_KEY_HERE")  # 🔐 Replace with your actual key
index = pc.Index("relbot")

# ---- Step 4: Prepare vectors ----
vectors = []
for doc in documents:
    embedding = model.encode(doc["text"]).tolist()
    vector_id = str(uuid.uuid4())
    
    metadata = {
        "religion": doc["religion"],
        "topic": doc["topic"],
        "source_title": doc.get("source_title", ""),
        "source_citation": doc.get("source_citation", ""),
        "source_type": doc.get("source_type", ""),
        "source_url": doc.get("source_url", ""),
        "year": doc.get("year", "")
    }
    
    vectors.append({
        "id": vector_id,
        "values": embedding,
        "metadata": metadata
    })

# ---- Step 5: Upsert in batches ----
namespace = "religion-views"
batch_size = 50

for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.upsert(vectors=batch, namespace=namespace)
    time.sleep(0.5)  # slight pause to avoid rate limits

print(f"✅ Uploaded {len(vectors)} vectors to namespace '{namespace}' in 'relbot'")
