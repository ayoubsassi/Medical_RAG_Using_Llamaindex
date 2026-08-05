from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


documents = SimpleDirectoryReader("./data").load_data()
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents[50:55])  # just first 5 pages, fast

print("Number of nodes:", len(nodes))
print("First node text:", nodes[0].text[:])
print("Second node text:", nodes[1].text[:])
print("Third node text:", nodes[2].text[:])

print("---")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec = embed_model.get_text_embedding(nodes[0].text)
print("Its embedding (first 10):", vec[:10])
print("Dimension:", len(vec))

print("Node 0 page:", nodes[0].metadata.get("page_label"))
print("Node 1 page:", nodes[1].metadata.get("page_label"))


from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

long_text = ". ".join([f"Sentence number {i} discussing health and medicine and the body" for i in range(80)]) + "."
doc = Document(text=long_text)

splitter = SentenceSplitter(chunk_size=50, chunk_overlap=15)
nodes = splitter.get_nodes_from_documents([doc])

print("Chunks:", len(nodes))
print("\nEND of chunk 0:\n", repr(nodes[0].text[-150:]))
print("\nSTART of chunk 1:\n", repr(nodes[1].text[:150]))