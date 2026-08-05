import os
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
import qdrant_client
from dotenv import load_dotenv

load_dotenv()   # reads .env into os.environ


#client = qdrant_client.QdrantClient(location=":memory:")  
client = qdrant_client.QdrantClient(   
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
    )  

vector_store = QdrantVectorStore(client=client, collection_name="test_store")

#Data loading
documents = SimpleDirectoryReader("./data").load_data()

"""from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (PDFReader, PyMuPDFReader)
# PDF Reader with `SimpleDirectoryReader`
parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()""" 
#this commented code and the one above is almost same but here we can customize our parser check this link for more details: https://llamahub.ai/l/readers/llama-index-readers-file?from=readers

"""print(type(documents))
print(len(documents))
print(repr(documents[50].text[:1500]))
print(documents[50].metadata)
"""
#Embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Creating chunks and doing the indexing
# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        #TitleExtractor(), #for metadata we can use if we have poor performance 
        embed_model,
    ],
    vector_store=vector_store,
)

# Ingest directly into a vector db
pipeline.run(documents=documents, show_progress=True)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

print("Points stored in Qdrant:", client.get_collection("test_store").points_count)
print("Ingestion complete.")