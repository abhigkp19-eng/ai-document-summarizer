from embedding_utils4 import get_embedding, cosine_similarity
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import PyPDF2

# Load env
load_dotenv(override=True)

# Azure client (for chat)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

chat_model = os.getenv("DEPLOYMENT_NAME")

# Step 1: Extract PDF text
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Chunking
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# Step 3: Load PDF
pdf_text = extract_text_from_pdf("Abhishek_7703806399.pdf")
chunks = chunk_text(pdf_text)

print(f"Total chunks: {len(chunks)}")

# Step 4: Embed chunks
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# Step 5: Ask question
query = input("\nAsk a question from PDF: ")
query_embedding = get_embedding(query)

# Step 6: Get top 3 relevant chunks
results = []

for i, emb in enumerate(chunk_embeddings):
    score = cosine_similarity(query_embedding, emb)
    results.append((chunks[i], score))

results = sorted(results, key=lambda x: x[1], reverse=True)
top_chunks = [chunk for chunk, score in results[:3]]

# Step 7: Send to Chat Model
context = "\n\n".join(top_chunks)

response = client.chat.completions.create(
    model=chat_model,
    messages=[
        {
            "role": "system",
            "content": "Answer only from the provided context."
        },
        {
            "role": "user",
            "content": f"""
            Context:
            {context}

            Question:
            {query}
            """
        }
    ]
)

# Step 8: Output
print("\n🤖 Answer:\n")
print(response.choices[0].message.content)