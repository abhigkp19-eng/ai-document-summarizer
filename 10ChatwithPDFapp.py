import streamlit as st
import os
import PyPDF2
from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np

load_dotenv(override=True)

# Azure client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

chat_model = os.getenv("DEPLOYMENT_NAME")
embedding_model = os.getenv("EMBEDDING_DEPLOYMENT")

# -------- Functions -------- #

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text):
    response = client.embeddings.create(
        model=embedding_model,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------- UI -------- #

st.title("📄 Chat with Your PDF (GenAI App)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)

    st.success(f"PDF loaded! Total chunks: {len(chunks)}")

    # Embed once
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    query = st.text_input("Ask a question:")

    if query:
        query_embedding = get_embedding(query)

        scores = []
        for i, emb in enumerate(chunk_embeddings):
            score = cosine_similarity(query_embedding, emb)
            scores.append((chunks[i], score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, score in scores[:3]]

        context = "\n\n".join(top_chunks)

        response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "Answer from the context only."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{query}"}
            ]
        )

        st.subheader("🤖 Answer")
        st.write(response.choices[0].message.content)