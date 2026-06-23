from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Settings
from config import API_KEY

Settings.embed_model = OpenAIEmbedding(
    embed_dim=1536,
    api_key=API_KEY,
    api_base="https://openrouter.ai/api/v1"
)
#storage_context = StorageContext.from_defaults(persist_dir="data/index")

reader = SimpleDirectoryReader("data/docs")
documents = reader.load_data()

index = VectorStoreIndex.from_documents(
    documents,
    #storage_context=storage_context,
    show_progress=True,
)

index.storage_context.persist(persist_dir="data/index")