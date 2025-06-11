from ollama import ollama


def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke an AI model given a system and user message.
    Replace this if you want to use a different AI model.
    """

    deepseek_client = DeepSeek() # Use ollama here to reference local DeepSeek AI model
    response = deepseek_client.chat.completions.create(
        model="deepseek-r1:8b",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content