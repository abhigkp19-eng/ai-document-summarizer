from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment = os.getenv("DEPLOYMENT_NAME")


def sql_to_english(sql_query):
    prompt = f"""
    Convert the following SQL query into simple English:

    SQL:
    {sql_query}

    Answer:
    """
    response = client.chat.completions.create(
        model=deployment,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def english_to_sql(user_input):
    prompt = f"""
    Convert the following English question into SQL query.
    Database schema:
    employees(id, name, department, salary)

    ONLY return SQL query.

    Question:
    {user_input}

    SQL:
    """
    response = client.chat.completions.create(
        model=deployment,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content