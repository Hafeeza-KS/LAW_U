import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import chromadb

# Load environment variables
load_dotenv()
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("legal_knowledge")

# Function to retrieve context
def retrieve_context(query, n_results=4, debug=False):
    # Query returns documents and metadata
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']  # may vary by chromadb version
    )

    docs = []
    metadatas = []
    if results.get("documents"):
        # results["documents"] is a list of lists
        for doc_list, meta_list in zip(results["documents"], results.get("metadatas", [])):
            for d, m in zip(doc_list, meta_list):
                docs.append(d)
                metadatas.append(m)

    if debug:
        print("=== Retrieved docs ===")
        for i, (d, m) in enumerate(zip(docs, metadatas)):
            print(f"#{i+1} distance: {results.get('distances', [[]])[0][i] if results.get('distances') else 'N/A'}")
            print("meta:", m)
            print("text:", d[:400].replace("\n"," ") + ("..." if len(d) > 400 else ""))
            print("----")

    if not docs:
        return "No relevant information found.", []
    # join with separators so the model can see boundaries
    joined = "\n\n---\n\n".join(docs)
    return joined, metadatas


# Function to generate AI response
def generate_response(query):
    # Build dynamic conversation memory
    chat_history = "\n".join([
        f"{sender}: {msg}" for sender, msg in st.session_state.chat_history[-6:]
    ])  # last 6 turns

    context = retrieve_context(query)

    prompt = f"""
    You are a knowledgeable and conversational AI legal assistant.
    Your specialization is in Indian women's legal rights.
    Use the context below and the conversation so far to respond naturally and accurately.

    --- Previous conversation ---
    {chat_history}

    --- Knowledge base context ---
    {context}

    --- User Query ---
    {query}

    Your task:
    - If the user asks for a shorter answer, summarize concisely (2–3 lines).
    - If the user asks for details, give a structured explanation.
    - Always maintain politeness and clarity.
    - End with: "Note: This information is for general awareness and not legal advice."
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # More powerful model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )

    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="Women's Legal Rights Chatbot", page_icon="⚖️", layout="centered")

st.title("⚖️ LAW-U – Legal Assistance for Indian Women’s Rights")
st.markdown("Ask any question related to women's legal rights in India.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)

# ✅ Chat input at bottom (fix)
if user_input := st.chat_input("Type your question here..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(("You", user_input))
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_response(user_input)
            st.markdown(answer)
            st.session_state.chat_history.append(("Bot", answer))

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit + Groq API")
