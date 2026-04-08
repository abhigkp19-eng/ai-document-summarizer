import os
from dotenv import load_dotenv

load_dotenv()

print("API KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
print("ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("DEPLOYMENT:", os.getenv("DEPLOYMENT_NAME"))


import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.chat.completions.create(
    model=os.getenv("DEPLOYMENT_NAME"),
    messages=[
        {"role": "user", "content": "Hello, are you working?"}
    ]
)

print(response.choices[0].message.content)


import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

load_dotenv()

async def main():
    print("DEPLOYMENT:", os.getenv("DEPLOYMENT_NAME"))

    kernel = Kernel()

    # ✅ FIXED CONFIG
    chat_service = AzureChatCompletion(
        service_id="chat",
        deployment_name=os.getenv("DEPLOYMENT_NAME"),  # my-gpt-model
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"   # 🔥 VERY IMPORTANT
    )

    kernel.add_service(chat_service)

    prompt = "Summarize: Semantic Kernel is a Microsoft framework for AI apps."

    # ✅ FIX: await
    result = await kernel.invoke_prompt(prompt)

    print("\n✅ Summary:\n", result)

asyncio.run(main())