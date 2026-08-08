import os
from dotenv import load_dotenv
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from IPython.display import display, Markdown
from llama_index.llms.google_genai import GoogleGenAI


load_dotenv()

#client = qdrant_client.QdrantClient(location=":memory:")  
client = qdrant_client.QdrantClient(   
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
    )  

vector_store = QdrantVectorStore(
                                    client=client, 
                                    collection_name="test_store_hybrid",
                                    enable_hybrid=True,
                                    fastembed_sparse_model="Qdrant/bm25"
                                    )

# 3. Load the SAME embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Build the index from the already-populated store
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# 5. Retrieve
retriever = index.as_retriever(similarity_top_k=2)
query="How is diabetes treated?"
results = retriever.retrieve(query)

print('Query: ',query)
for r in results:
    print(r.score, "| page", r.node.metadata.get("page_label"))
    print(r.node.text[:])
    print("---")

# Llm API call
llm = GoogleGenAI(
    model="gemini-3.6-flash",   # check current model name in AI Studio
    api_key=os.environ["GOOGLE_API_KEY"],
)


# retrieve 2 sparse, 2 dense, and filter down to 3 total hybrid results
query_engine = index.as_query_engine(
    llm=llm,
    vector_store_query_mode="hybrid",
    sparse_top_k=2,
    similarity_top_k=2,
    hybrid_top_k=3,
)

query="How is diabetes treated?"
print('Query: ',query)
response = query_engine.query(query)
print(response)
