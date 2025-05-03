import os
import uuid
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # not used directly, but fine to keep
HF_API_TOKEN = os.getenv("HUGGING")

# -------------------------
# Initialize Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("relbot")
namespace = "religion-views"


# -------------------------
# Hugging Face API config
# -------------------------
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# -------------------------
# Embed using HF API
# -------------------------
def get_hf_embedding(text):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    if response.status_code == 200:
        embedding = response.json()
        # Flatten if nested
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
    filter = {}
    if religion_filter and religion_filter != "All":
        filter = {"religion": {"$eq": religion_filter}}
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filter if filter else None
    )
    return results.matches

def build_prompt(user_question, retrieved_chunks):
    context_texts = []
    for match in retrieved_chunks:
        title = match.metadata.get("source_title", "Unknown Source")
        text = match.metadata.get("text", "")
        context_texts.append(f"{title}: {text}")
    context_str = "\n\n".join(context_texts)
    prompt = (
        "You are a helpful assistant knowledgeable about religious teachings.\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {user_question}\n"
        "Answer in detail, drawing only from the context. "
        "Provide inline citations using the source titles for each fact."
    )
    return prompt

def generate_answer_from_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.7}
    }
    resp = requests.post(HF_API_URL.replace("feature-extraction", "text-generation"), headers=HF_HEADERS, json=payload)
    try:
        result = resp.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "")
        return result.get("generated_text", result)
    except:
        return "Error generating answer."

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

# Tab setup
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
                prompt = build_prompt(question, matches)
                answer = generate_answer_from_llm(prompt)

                st.markdown("### Answer:")
                st.write(answer)

                if show_sources:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for match in matches:
                        title = match.metadata.get("source_title", "Unknown Source")
                        url = match.metadata.get("source_url", "")
                        st.markdown(f"- *{title}* – [{url}]({url})")

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
