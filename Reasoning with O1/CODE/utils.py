import os
import json
import requests
import base64
from helper import get_openai_api_key

def o1_tools(messages, model, tools):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    data = {
        "model": model,
        "messages": messages,
        "tools": tools
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response.json()

def o1_vision(file_path, prompt, model):
    with open(file_path, 'rb') as file:
        base64_image = base64.b64encode(file.read()).decode('utf-8')

    url = f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_openai_api_key()}"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "auto"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response.json()