import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEndpoint

# Load env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE")
HF_API_TOKEN = os.getenv("HUGGING")

# Setup Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("relbot")
namespace = "religion-views"

# Hugging Face embedding API
EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# LangChain LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    huggingfacehub_api_token=HF_API_TOKEN,
    max_new_tokens=400,
    temperature=0.7
)


# Instruction mapping
def get_prompt_instructions(mode):
    return {
        "ask": "You are a helpful assistant knowledgeable about religious teachings.\nAnswer the question using only the context below with inline citations.",
        "tension": "You are a religious advisor helping people resolve real-world moral tensions. Use the context below to offer faith-based insight.",
        "ethics": "You are an ethics coach offering value-based advice. Use the religious context to reflect on the user's personal dilemma.",
        "interfaith": "You are comparing religious perspectives. Use the context to summarize similarities and differences across faiths.",
        "guided": "You are an educational guide. Use the context to teach structured insights on the topic and offer reflection points."
    }.get(mode, "You are a helpful assistant. Use the context to answer the question.")

# Get embeddings
def get_hf_embedding(text):
    print(f"🔎 Getting embedding for: {text}")
    response = requests.post(EMBEDDING_URL, headers=HF_HEADERS, json={"inputs": text})
    if response.status_code == 200:
        data = response.json()
        return data[0] if isinstance(data, list) and isinstance(data[0], list) else data
    else:
        print("❌ Embedding error:", response.text)
        return None

# Query Pinecone
def retrieve_relevant_chunks(embedding, religion_filter="All", top_k=5):
    filter = {"religion": {"$eq": religion_filter}} if religion_filter != "All" else {}
    print(f"📦 Querying Pinecone with filter: {filter}")
    results = index.query(
        vector=embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filter
    )
    print(f"✅ Retrieved {len(results.matches)} matches")
    return results.matches

# Build prompt and citation map
def build_prompt_with_citations(user_question, matches, mode):
    context_lines = []
    citation_map = {}

    for i, match in enumerate(matches):
        title = match.metadata.get("source_title", "Unknown Source")
        url = match.metadata.get("source_url", "")
        text = match.metadata.get("text", "")
        tag = f"[{i+1}]"
        context_lines.append(f"{tag} {text}")
        citation_map[tag] = f"{title} ({url})"

    context_str = "\n\n".join(context_lines)
    instruction = get_prompt_instructions(mode)

    prompt = (
        f"{instruction}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {user_question}\n"
        "Answer in detail, citing sources using [1], [2], etc. Inline citations only from context."
    )
    return prompt, citation_map

# Generate with LangChain LLM
def generate_answer_from_llm(prompt):
    print("🧠 Sending prompt to HuggingFace LLM...")
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response
    except Exception as e:
        print("❌ LLM generation error:", e)
        return "Error generating response."

# === MAIN FLOW ===
if __name__ == "__main__":
    user_question = "What does Islam say about wealth and giving?"
    selected_religion = "Islam"
    mode = "ask"

    embedding = get_hf_embedding(user_question)
    if embedding:
        matches = retrieve_relevant_chunks(embedding, religion_filter=selected_religion)
        prompt, citations = build_prompt_with_citations(user_question, matches, mode)

        print("\n📄 Prompt Sent to LLM:\n", prompt)
        print("\n🔖 Citation Map:")
        for tag, source in citations.items():
            print(f"  {tag}: {source}")

        answer = generate_answer_from_llm(prompt)
        print("\n🧠 Final Answer:\n", answer)
    else:
        print("⚠️ Failed to get embedding.")
