from rag_setup import get_retriever, get_query_engine

query="What are the symptoms of malaria?"
print('Query: ',query)

retriever = get_retriever()
results = retriever.retrieve(query)

for r in results:
    print(r.score, "| page", r.node.metadata.get("page_label"))
    print(r.node.text[:])
    print("---")

query_engine = get_query_engine()
print(query_engine.query(query))