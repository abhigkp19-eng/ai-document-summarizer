from embedding_utils4 import get_embedding, cosine_similarity

# Step 1: Data
documents = [
    "Artificial Intelligence is transforming the world",
    "Machine learning is a subset of AI",
    "Deep learning improves AI systems",
    "I love playing cricket",
    "Football is a popular sport"
]

# Step 2: Precompute embeddings
doc_embeddings = [get_embedding(doc) for doc in documents]

# Step 3: User input
query = input("Enter your search query: ")
query_embedding = get_embedding(query)

# Step 4: Store scores
results = []

for i, doc_emb in enumerate(doc_embeddings):
    score = cosine_similarity(query_embedding, doc_emb)
    results.append((documents[i], score))

# Step 5: Sort by score (descending)
results = sorted(results, key=lambda x: x[1], reverse=True)

# Step 6: Show Top 3
print("\n🔍 Top 3 Matches:\n")

for doc, score in results[:3]:
    print(f"{round(score, 3)} → {doc}")