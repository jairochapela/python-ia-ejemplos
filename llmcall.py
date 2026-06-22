import requests
import json
from config import API_KEY

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "http://localhost", # Optional. Site URL for rankings on openrouter.ai.
    "X-OpenRouter-Title": "LLM Call PoC", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "~openai/gpt-latest",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)

print(response.json())