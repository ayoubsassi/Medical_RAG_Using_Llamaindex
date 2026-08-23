from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from utils import get_embed_model, get_vector_store, load_doc
import config


def main():
    documents = load_doc()
    embed_model = get_embed_model()
    client, vector_store = get_vector_store()
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP),
            embed_model,
        ],
        vector_store=vector_store,
    )
    pipeline.run(documents=documents, show_progress=True)
    print("Points stored in Qdrant:", client.get_collection(config.COLLECTION_NAME).points_count)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()