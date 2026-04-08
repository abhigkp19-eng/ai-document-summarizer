import streamlit as st
import PyPDF2
import os
import faiss
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

# -------------------------------
# Load Environment
# -------------------------------
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment = os.getenv("DEPLOYMENT_NAME")
embedding_model = os.getenv("EMBEDDING_DEPLOYMENT")

# -------------------------------
# STEP 4: Extract + Chunk
# -------------------------------
def extract_text(UPS_QA_200):
    reader = PyPDF2.PdfReader(UPS_QA_200)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# -------------------------------
# STEP 5: Embeddings
# -------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model=embedding_model,
        input=text
    )
    return response.data[0].embedding

# -------------------------------
# STEP 6: Vector Store (FAISS)
# -------------------------------
def create_vector_store(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings).astype("float32"))

    return index

# -------------------------------
# STEP 7: Search
# -------------------------------
def search(query, index, chunks, k=3):
    query_embedding = get_embedding(query)

    D, I = index.search(
        np.array([query_embedding]).astype("float32"), k
    )

    results = [chunks[i] for i in I[0]]
    return results

# -------------------------------
# STEP 8: Ask LLM (RAG)
# -------------------------------
def ask_llm(query, context):
    prompt = f"""
    Answer based only on the context below:

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful company assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

# -------------------------------
# UI (Streamlit)
# -------------------------------
st.title("🏢 AI Company Knowledge Chatbot")

uploaded_file = st.file_uploader("Upload Company PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully")

    text = extract_text(uploaded_file)
    chunks = chunk_text(text)

    with st.spinner("Creating embeddings..."):
        index = create_vector_store(chunks)

    st.success("✅ Ready to ask questions!")

    query = st.text_input("Ask your question")

    if query:
        with st.spinner("Searching + Generating answer..."):
            results = search(query, index, chunks)
            context = " ".join(results)

            answer = ask_llm(query, context)

            st.subheader("💡 Answer")
            st.write(answer)