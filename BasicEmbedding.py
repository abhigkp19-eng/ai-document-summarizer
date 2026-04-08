import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT")

print("Using deployment:", embedding_deployment)

response = client.embeddings.create(
    model=embedding_deployment,
    input="rtificial Intelligence is transforming the world"
)

embedding = response.data[0].embedding

print("Vector length:", len(embedding))
print("First 10 values:", embedding[:10])