import time, json, ast
import pandas as pd
from utils import get_query_engine

def main():
    # load the test questions
    testset = pd.read_csv("testset.csv")

    query_engine = get_query_engine()

    results = []
    for i, row in testset.iterrows():
        question = row["user_input"]
        print(f"[{i+1}/{len(testset)}] {question}...")

        response = query_engine.query(question)

        results.append({
            "user_input": question,
            "response": str(response),                                    # your RAG's answer
            "retrieved_contexts": [n.node.text for n in response.source_nodes],  # what it retrieved
            "reference": row["reference"],                                # ground-truth answer from testset
            "reference_contexts": ast.literal_eval(row["reference_contexts"]),  # parsed to real list

        })

    # save so we never have to regenerate (protects the quota spent)
    with open("rag_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} RAG outputs to rag_outputs.json")
    time.sleep(60)   # ~12 calls/min, under the 15 limit

if __name__ == "__main__":
    main()
    