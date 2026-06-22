import requests
import json
from config import API_KEY

response = requests.post(
  url="https://openrouter.ai/api/v1/embeddings",
  headers={
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost", # Optional. Site URL for rankings on openrouter.ai.
    "X-OpenRouter-Title": "Embeddings PoC", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "openai/text-embedding-3-small",
    "input": "The quick brown fox jumps over the lazy dog",
    "dimensions": 1536
  })
)

print(response.json())