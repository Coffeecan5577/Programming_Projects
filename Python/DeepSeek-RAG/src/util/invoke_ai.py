import os
from ollama_remote_client import OllamaRemoteClient

def invoke_ai(
    ollama_url: str | None,
    ollama_model: str | None,
    system_message: str,
    user_message: str,
) -> str:

    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    if ollama_model is None:
        ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

    LLM = OllamaRemoteClient(ollama_url, ollama_model)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    reply = LLM.chat(messages=messages, stream=False, temperature=0)
    return reply