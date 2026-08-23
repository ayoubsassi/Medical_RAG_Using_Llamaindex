import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import VectorStoreIndex
import config

def get_embed_model():
    return HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)

def get_llm():
    return GoogleGenAI(model=config.GEMINI_MODEL, api_key=config.GOOGLE_API_KEY)

def get_vector_store():
    client = qdrant_client.QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )
    return client, vector_store


def get_index():
    embed_model = get_embed_model()
    _, vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def get_retriever():
    return get_index().as_retriever(
        vector_store_query_mode="hybrid",
        sparse_top_k=config.SPARSE_TOP_K,
        similarity_top_k=config.SIMILARITY_TOP_K,
        hybrid_top_k=config.HYBRID_TOP_K,
        )
    

def get_query_engine():
    return get_index().as_query_engine(
        llm=get_llm(),
        vector_store_query_mode="hybrid",
        sparse_top_k=config.SPARSE_TOP_K,
        similarity_top_k=config.SIMILARITY_TOP_K,
        hybrid_top_k=config.HYBRID_TOP_K,
    )
    