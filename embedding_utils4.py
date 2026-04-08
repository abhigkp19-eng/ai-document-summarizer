import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

embedding_model = os.getenv("EMBEDDING_DEPLOYMENT")


def get_embedding(text):
    response = client.embeddings.create(
        model=embedding_model,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    text1 = "Artificial Intelligence is transforming the world"
    text2 = "AI is changing the future"

    text3 = "Artificial Intelligence  "
    text4 = "AI"
    
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    emb3 = get_embedding(text3)
    emb4 = get_embedding(text4)

    print("Vector length:", len(emb1))

    score = cosine_similarity(emb1, emb2)
    
    score1 = cosine_similarity(emb3, emb4)
    
    
    print("Similarity Score:", score)
    print("Similarity Score 2:", score1)
    