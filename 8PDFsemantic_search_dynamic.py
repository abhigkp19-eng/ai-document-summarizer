from embedding_utils4 import get_embedding, cosine_similarity
import PyPDF2

# Step 1: Read PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        
        for page in reader.pages:
            text += page.extract_text()
    
    return text

# Step 2: Chunk text (important for large PDFs)
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

# Step 3: Load PDF
pdf_text = extract_text_from_pdf("Abhishek Srivastava reume Analyser.pdf")

# Step 4: Chunk it
chunks = chunk_text(pdf_text)

print(f"Total chunks: {len(chunks)}")

# Step 5: Embed all chunks
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# Step 6: User query
query = input("Ask something from PDF: ")
query_embedding = get_embedding(query)

# Step 7: Find best match
results = []

for i, chunk_emb in enumerate(chunk_embeddings):
    score = cosine_similarity(query_embedding, chunk_emb)
    results.append((chunks[i], score))

# Step 8: Sort results
results = sorted(results, key=lambda x: x[1], reverse=True)

# Step 9: Show top 3 results
print("\n🔍 Top Matches from PDF:\n")

for chunk, score in results[:3]:
    print(f"Score: {round(score, 3)}")
    print(chunk[:200])  # print first 200 chars
    print("-" * 50)