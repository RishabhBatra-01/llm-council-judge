import json
import os

import requests
from dotenv import load_dotenv


# 1. Load the .env file and read the API key from the environment.
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")


# 2. OpenRouter chat completions endpoint.
url = "https://openrouter.ai/api/v1/chat/completions"


# 3. Build the request headers.
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}


# 4. Build the request body.
body = {
    "model": "inclusionai/ling-3.0-flash-fin:free",
    "messages": [
        {
            "role": "user",
            "content": "Say hello in exactly five words.",
        }
    ],
}


# 5. Send the request and inspect the full response.
response = requests.post(url, headers=headers, json=body)
response.raise_for_status()

response_data = response.json()

# Print the entire response first so you can inspect its structure.
print(json.dumps(response_data, indent=2))

# Dig into the nested response to get the model's answer.
answer = response_data["choices"][0]["message"]["content"]
print(answer)
