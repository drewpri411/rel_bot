import json
import os
import uuid
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ---- Load ENV ----
load_dotenv()
pinecone_key = os.getenv("PINECONE")

# ---- Init Pinecone ----
pc = Pinecone(api_key=pinecone_key)
index_name = "relbot"

# ---- Delete the index if it exists ----
if index_name in [i.name for i in pc.list_indexes()]:
    print(f"🗑️ Deleting index '{index_name}'...")
    pc.delete_index(index_name)
    time.sleep(2)

# ---- Recreate index ----
print(f"⚙️ Recreating index '{index_name}'...")
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# ---- Wait for index to be ready ----
print("⏳ Waiting for index to be ready...")
while True:
    status = pc.describe_index(index_name).status
    if status["ready"]:
        print("✅ Index is ready.")
        break
    time.sleep(1)

index = pc.Index(index_name)

# ---- Load data ----
with open("data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# ---- Load embedding model ----
model = SentenceTransformer('all-MiniLM-L6-v2')  # dim = 384

# ---- Embed and prepare vectors ----
vectors = []
for doc in documents:
    embedding = model.encode(doc["text"]).tolist()
    vector_id = str(uuid.uuid4())

    metadata = {
        "religion": doc["religion"],
        "topic": doc["topic"],
        "text": doc["text"],  # 👈 ADD THIS LINE
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

# ---- Upsert in batches ----
namespace = "religion-views"
batch_size = 50

print(f"📤 Uploading {len(vectors)} vectors to namespace '{namespace}'...")
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(vectors=batch, namespace=namespace)
    time.sleep(0.5)

print("✅ All vectors uploaded successfully!")
