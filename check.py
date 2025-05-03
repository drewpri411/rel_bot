import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone

# === Load environment variables ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE")
HF_API_TOKEN = os.getenv("HUGGING")

# === Pinecone Setup ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("relbot")
namespace = "religion-views"

# === Hugging Face Setup ===
EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
LLM_URL = "https://api-inference.huggingface.co/pipeline/text-generation/mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# === Helper: Instruction Presets ===
def get_prompt_instructions(mode):
    return {
        "ask": "You are a helpful assistant knowledgeable about religious teachings.\nAnswer the question using only the context below with inline citations.",
        "tension": "You are a religious advisor helping people resolve real-world moral tensions. Use the context below to offer faith-based insight.",
        "ethics": "You are an ethics coach offering value-based advice. Use the religious context to reflect on the user's personal dilemma.",
        "interfaith": "You are comparing religious perspectives. Use the context to summarize similarities and differences across faiths.",
        "guided": "You are an educational guide. Use the context to teach structured insights on the topic and offer reflection points."
    }.get(mode, "You are a helpful assistant. Use the context to answer the question.")

# === Step 1: Get Embedding ===
def get_hf_embedding(text):
    print(f"🔎 Getting embedding for: {text}")
    resp = requests.post(EMBEDDING_URL, headers=HF_HEADERS, json={"inputs": text})
    if resp.status_code == 200:
        data = resp.json()
        return data[0] if isinstance(data, list) and isinstance(data[0], list) else data
    else:
        print("❌ Embedding Error:", resp.text)
        return None

# === Step 2: Retrieve Relevant Chunks ===
def retrieve_relevant_chunks(embedding, religion_filter="Islam", top_k=5):
    filter = {"religion": {"$eq": religion_filter}} if religion_filter != "All" else {}
    print(f"📦 Querying Pinecone with filter: {filter}")
    results = index.query(
        vector=embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filter or None
    )
    print(f"✅ Retrieved {len(results.matches)} matches")
    return results.matches

# === Step 3: Build Prompt with Citations ===
def build_prompt_with_citations(user_question, matches):
    context_lines = []
    citation_map = {}

    for i, match in enumerate(matches):
        title = match.metadata.get("source_title", "Unknown Source")
        url = match.metadata.get("source_url", "")
        text = match.metadata.get("text", "")
        citation_tag = f"[{i+1}]"
        source_citation = f"{title} ({url})"
        context_lines.append(f"{citation_tag} {text}")
        citation_map[citation_tag] = source_citation

    context_str = "\n\n".join(context_lines)
    prompt = (
        f"Context:\n{context_str}\n\n"
        f"Question: {user_question}\n"
        "Answer in detail, citing sources using [1], [2], etc. Inline citations only from context."
    )
    return prompt, citation_map

# === Step 4: Generate Answer ===
def generate_answer_from_llm(prompt):
    print("🧠 Sending prompt to LLM...")
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 400, "temperature": 0.7}}
    resp = requests.post(LLM_URL, headers=HF_HEADERS, json=payload)
    try:
        result = resp.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "")
        return result.get("generated_text", result)
    except Exception as e:
        print("❌ LLM Error:", e)
        return "Error generating response."

# === MAIN DEBUG FLOW ===
if __name__ == "__main__":
    user_question = "What does Islam say about wealth and giving?"
    selected_religion = "Islam"
    mode = "ask"  # Can be: ask, tension, ethics, interfaith, guided

    embedding = get_hf_embedding(user_question)
    if embedding:
        matches = retrieve_relevant_chunks(embedding, religion_filter=selected_religion)
        prompt_core, citation_map = build_prompt_with_citations(user_question, matches)
        instruction = get_prompt_instructions(mode)
        final_prompt = f"{instruction}\n\n{prompt_core}"

        print("\n📄 Prompt Sent to LLM:\n", final_prompt)
        print("\n🔖 Citation Map:")
        for tag, source in citation_map.items():
            print(f"  {tag}: {source}")

        answer = generate_answer_from_llm(final_prompt)
        print("\n🧠 Answer from LLM:\n", answer)
    else:
        print("⚠️ Failed to embed user input.")
