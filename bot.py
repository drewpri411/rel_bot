import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEndpoint

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
HF_API_TOKEN = os.getenv("HUGGING")

# -------------------------
# Initialize Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("relbot")
namespace = "religion-views"

# -------------------------
# Hugging Face Embedding API config
# -------------------------
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# -------------------------
# Initialize HuggingFace LLM via LangChain
# -------------------------
# LangChain LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    huggingfacehub_api_token=HF_API_TOKEN,
    max_new_tokens=400,
    temperature=0.7
)

# -------------------------
# Mode-specific instructions
# -------------------------
def get_prompt_instructions(mode):
    return {
        "Ask a Question": "You are a helpful assistant knowledgeable about religious teachings.\nAnswer the question using only the context below with inline citations.",
        "Tension Explorer": "You are a religious advisor helping people resolve real-world moral tensions. Use the context below to offer faith-based insight.",
        "Ethics Coach": "You are an ethics coach offering value-based advice. Use the religious context to reflect on the user's personal dilemma.",
        "Interfaith Harmony": "You are comparing religious perspectives. Use the context to summarize similarities and differences across faiths.",
        "Guided Exploration": "You are an educational guide. Use the context to teach structured insights on the topic and offer reflection points."
    }.get(mode, "You are a helpful assistant. Use the context to answer the question.")

# -------------------------
# Embed using HF API
# -------------------------
def get_hf_embedding(text):
    response = requests.post(HF_EMBED_URL, headers=HF_HEADERS, json={"inputs": text})
    if response.status_code == 200:
        embedding = response.json()
        if isinstance(embedding, list) and isinstance(embedding[0], list):
            return embedding[0]
        return embedding
    else:
        st.error("❌ Failed to fetch embedding from Hugging Face API.")
        return None

# -------------------------
# Helper functions
# -------------------------
def retrieve_relevant_chunks(query_embedding, religion_filter=None, top_k=5):
    filter = {"religion": {"$eq": religion_filter}} if religion_filter and religion_filter != "All" else {}
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filter if filter else None
    )
    return results.matches

def build_prompt(user_question, retrieved_chunks, mode):
    context_texts = []
    citation_map = {}
    for i, match in enumerate(retrieved_chunks):
        idx = i + 1
        title = match.metadata.get("source_title", "Unknown Source")
        url = match.metadata.get("source_url", "")
        text = match.metadata.get("text", "")
        context_texts.append(f"[{idx}] {text}")
        citation_map[str(idx)] = f"{title} ({url})"
    context_str = "\n\n".join(context_texts)
    instruction = get_prompt_instructions(mode)
    prompt = (
        f"{instruction}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {user_question}\n"
        "Answer in detail, citing sources using [1], [2], etc. Inline citations only from context."
    )
    return prompt, citation_map

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Interfaith Chatbot", layout="wide")
st.title("🤖 Interfaith Chatbot: Faith & Money in a Secular World")

st.markdown("""
This chatbot helps you explore how different religions view **money, wealth, and ethical living** in today's secular society. It's powered by a custom knowledge base of religious texts, expert commentary, and real-world stories, and gives **source-backed, religion-specific answers**.

---

### 🧭 How to Use This Chatbot
1. Navigate through the tabs to explore different features.
2. Select a religion or view all.
3. Type your question or select a prompt.
4. Press **Submit** to see your answer with citations.
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tabs = st.tabs(["Ask a Question", "Tension Explorer", "Ethics Coach", "Interfaith Harmony", "Guided Exploration"])
tab_names = ["Ask a Question", "Tension Explorer", "Ethics Coach", "Interfaith Harmony", "Guided Exploration"]

instructions = {
    "Ask a Question": "Use this tab to ask any open-ended question about wealth, ethics, or lifestyle choices. Get a response grounded in teachings from your selected religion, or compare across all faiths.",
    "Tension Explorer": "Explore real-life dilemmas people face when religious teachings meet modern economic realities.",
    "Ethics Coach": "Ask direct 'What should I do?' questions and receive thoughtful perspectives based on moral guidance.",
    "Interfaith Harmony": "See what different religions share in common around money, charity, and justice.",
    "Guided Exploration": "Take a journey through topics like 'Ethics at Work' or 'Faithful Finance'."
}

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(tab_names[i])
        st.info(instructions[tab_names[i]])
        religion = st.selectbox("Select Religion", ["All", "Christianity", "Islam", "Judaism", "Hinduism", "Buddhism"], key=f"religion_{i}")
        question = st.text_input("Your question:", key=f"question_{i}")
        show_sources = st.checkbox("Show full sources", key=f"sources_{i}")

        if st.button("Submit", key=f"submit_{i}") and question.strip():
            embed = get_hf_embedding(question)
            if embed:
                matches = retrieve_relevant_chunks(embed, religion_filter=religion)
                prompt, citation_map = build_prompt(question, matches, tab_names[i])
                answer = llm.invoke(prompt)
                answer = answer.content if hasattr(answer, "content") else answer

                st.markdown("### Answer:")
                st.write(answer)

                if show_sources:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for idx, source in citation_map.items():
                        st.markdown(f"[{idx}]: {source}")

                st.session_state.chat_history.append({
                    "tab": tab_names[i],
                    "question": question,
                    "answer": answer
                })

        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("**Session History:**")
            for entry in st.session_state.chat_history:
                if entry["tab"] == tab_names[i]:
                    st.markdown(f"**Q:** {entry['question']}")
                    st.markdown(f"**A:** {entry['answer']}")
