import os
import logging
from dotenv import load_dotenv
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.ERROR)

# Config
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BQ_DATASET = os.getenv("BQ_INSTANCE")
BQ_TABLE = os.getenv("BQ_TABLE")

def main():
    # Same embedding model used during indexing
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=PROJECT_ID,
        location=REGION
    )

    client = bigquery.Client(project=PROJECT_ID)

    query_text = "What is august?"
    print(f"\nQUERY: {query_text}\n" + "=" * 50)

    # Embed query once
    query_vector = embeddings.embed_query(query_text)

    strategies = ["character", "recursive", "token"]

    for strategy in strategies:
        print(f"\nSTRATEGY: {strategy.upper()}")

        sql = f"""
        SELECT
          content,
          COSINE_DISTANCE(embedding_vector, @q) AS distance
        FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE strategy = @strategy
        ORDER BY distance
        LIMIT 2
        """

        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("q", "FLOAT64", query_vector),
                    bigquery.ScalarQueryParameter("strategy", "STRING", strategy),
                ]
            ),
        )

        results = list(job)

        for i, row in enumerate(results):
            preview = row.content[:150].replace("\n", " ")
            print(f"  Result {i+1} (Distance: {row.distance:.4f}): {preview}...")

if __name__ == "__main__":
    main()
