from dotenv import load_dotenv
from ragas.testset import TestsetGenerator
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from utils import *

load_dotenv()

def main():

    # only need docs + llm + embeddings for GENERATION (no Qdrant/index here)
    documents = load_doc()
    subset = documents[487:490]   # small clean slice

    embed_model = get_embed_model()
    llm = get_llm()

    # wrap for RAGAS
    ragas_llm = LlamaIndexLLMWrapper(llm)
    ragas_emb = LlamaIndexEmbeddingsWrapper(embed_model)

    generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_emb)

    dataset = generator.generate_with_llamaindex_docs(subset, testset_size=3)

    df = dataset.to_pandas()
    print(df.head())
    #df.to_csv("testset.csv", index=False)  #for the first time only 
    df.to_csv("testset.csv", mode="a", header=False, index=False)
    print(f"Saved {len(df)} test questions to testset.csv")

if __name__ == "__main__":
    main()