from embedding_utils4 import get_embedding, cosine_similarity

# Step 1: Data (mini database)
documents = [
    "Artificial Intelligence is transforming the world",
    "Machine learning is a subset of AI",
    "Deep learning improves AI systems",
    "I love playing cricket",
    "Football is a popular sport"
]

# Step 2: Precompute embeddings
doc_embeddings = [get_embedding(doc) for doc in documents]

# Step 3: Take user input
query = input("Enter your search query: ")

query_embedding = get_embedding(query)

# Step 4: Find best match
best_score = -1
best_doc = ""

for i, doc_emb in enumerate(doc_embeddings):
    score = cosine_similarity(query_embedding, doc_emb)

    if score > best_score:
        best_score = score
        best_doc = documents[i]

# Step 5: Output
print("\n🔍 Best Match:")
print(best_doc)
print("Score:", round(best_score, 3))