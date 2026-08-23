# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# Qdrant
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = "test_store_hybrid"

# Models
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Retrieval settings
SPARSE_TOP_K = 2
SIMILARITY_TOP_K = 2
HYBRID_TOP_K = 3

# Paths
DATA_DIR = "./data"