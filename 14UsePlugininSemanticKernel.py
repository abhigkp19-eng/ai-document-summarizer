import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from math_plugin import MathPlugin

load_dotenv()

async def main():
    kernel = Kernel()

    # Add AI service
    kernel.add_service(
        AzureChatCompletion(
            service_id="chat",
            deployment_name=os.getenv("DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01"
        )
    )

    # 🔥 Add plugin
    kernel.add_plugin(MathPlugin(), plugin_name="math")

    # Ask AI to use plugin
    prompt = "What is 5 + 10?"

    result = await kernel.invoke_prompt(prompt)

    print("\n✅ Answer:\n", result)

asyncio.run(main())