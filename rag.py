from llama_index.llms.openrouter import OpenRouter
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from config import API_KEY

Settings.embed_model = OpenAIEmbedding(
    embed_dim=1536,
    api_key=API_KEY,
    api_base="https://openrouter.ai/api/v1"
)

storage_context = StorageContext.from_defaults(persist_dir="data/index")
index = load_index_from_storage(storage_context)
llm = OpenRouter(
    api_key=API_KEY,
    max_tokens=256,
    context_window=4096,
    model="openai/gpt-4o-mini",
)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("¿De qué trata el informe?")
print(response)