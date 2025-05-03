import requests
import streamlit as st
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEndpoint

# -------------------------
# Load environment variables
# -------------------------
PINECONE_API_KEY = st.secrets["PINECONE"]
PINECONE_ENV = st.secrets["PINECONE_ENVIRONMENT"]
HF_API_TOKEN = st.secrets["HUGGING"]

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
Welcome to the **🤖 Interfaith Chatbot: Faith & Money in a Secular World** — a user-friendly AI assistant that helps you explore what different religions say about **money, ethics, work, and wealth** in today’s world.

This chatbot is created for learners, seekers, and anyone interested in how spiritual traditions guide financial and ethical decisions. It draws on a carefully selected set of religious teachings, expert insights, and real-world examples across **Christianity, Islam, Judaism, Hinduism, and Buddhism**.

Unlike generic chatbots, this app has been trained on a **custom knowledge base** of high-quality sources. That means it’s not just guessing — it gives meaningful, faith-based answers because the right material was fed directly into its memory.

---

### 🌍 What This Chatbot Does
- Answers your questions with insights from real religious texts and commentary
- Lets you focus on a single religion or compare across multiple faiths
- Explains how money, charity, wealth, and ethical living are seen in each tradition
- Provides easy-to-read answers with **citations** linked to original sources

---

### ✨ How to Use It
1. Choose a **tab** based on what kind of help or answer you’re looking for.
2. Pick a **religion** or select **All** to compare across traditions.
3. Ask a question about **money, faith, ethics, or real-life dilemmas**.
4. Click **Submit** to get an answer backed by real sources.
5. Turn on **“Show full sources”** to view references used in your answer.

---

💬 You can also scroll down in any tab to see your **Session History** — the questions you’ve asked and the answers you’ve received.

Try it out and discover how faith and money connect in meaningful, thoughtful ways!
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Tabs setup
tabs = st.tabs(["Ask a Question", "Tension Explorer", "Ethics Coach", "Interfaith Harmony", "Guided Exploration"])
tab_names = ["Ask a Question", "Tension Explorer", "Ethics Coach", "Interfaith Harmony", "Guided Exploration"]

instructions = {
    "Ask a Question": "Use this tab to ask any **open-ended question** about money, ethics, or economic life. You can explore a single religion or compare all.\n\n**Purpose:** General exploration of faith teachings.\n**Example:** 'What does Buddhism say about property ownership?'",

    "Tension Explorer": "Use this tab to describe a **real-world dilemma** where faith meets financial or modern pressure.\n\n**Purpose:** Discover how different religions guide followers in tough situations.\n**Example:** 'Is it okay to earn bonuses from alcohol sales if I’m Hindu?'",

    "Ethics Coach": "Ask personal or moral questions like 'What should I do?' and receive thoughtful advice.\n\n**Purpose:** Get ethical and introspective support, not just doctrine.\n**Example:** 'I feel guilty about being rich — what would a Christian response be?'",

    "Interfaith Harmony": "Compare how various religions treat common financial themes.\n\n**Purpose:** Promote empathy by surfacing **shared values** across traditions.\n**Example:** 'Do all religions promote giving to the poor?'",

    "Guided Exploration": "Choose from prewritten learning journeys on topics like **'Ethics at Work'**, **'Charity & Simplicity'**, and more.\n\n**Purpose:** Structured learning with suggested questions and reflection prompts.\n**Example:** 'Walk through how religious values affect budgeting and generosity.'"
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
