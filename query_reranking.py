import os
import logging
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings

# Reranking Imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank
from langchain_google_community import BigQueryVectorStore
from BigQueryRetriever import BigQueryVectorRetriever

from google.cloud import bigquery

load_dotenv()
logging.basicConfig(level=logging.ERROR)

PROJECT_ID = "buildathon-485822"
REGION = "us-central1"
BQ_DATASET = "advanced_rag"
BQ_TABLE = "wikipedia_vectors"

# IMPORTANT: Target the recursive collection created in ingest_data.py
RANKING_LOCATION = "global"

def main():
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)

    bq_client = bigquery.Client(project=PROJECT_ID)

    # 1. Base Retriever (Vector Search) - Fetch top 10
    retriever = BigQueryVectorRetriever(
                    client=bq_client,
                    table="buildathon-485822.advanced_rag.wikipedia_vectors",
                    embeddings=embeddings,
                    strategy="recursive",
                    k=20
                )

    query = "What are the arts?"
    print(f"QUERY: {query}\n")

    # 2. Reranker - Select top 3 from the 10
    reranker = VertexAIRank(
        project_id=PROJECT_ID,
        location_id=RANKING_LOCATION,
        ranking_config="default_ranking_config",
        title_field="source",
        top_n=3
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever
    )

    # Execute
    try:
        reranked_docs = compression_retriever.invoke(query)

        if not reranked_docs:
            print("No documents returned. Check if the collection exists and is populated.")

        print(f"--- Top 3 Reranked Results ---")
        for i, doc in enumerate(reranked_docs):
            print(f"Result {i+1} (Score: {doc.metadata.get('relevance_score', 'N/A')}):")
            print(f"  {doc.page_content}...\n")
    except Exception as e:
        print(f"Error during reranking: {e}")

if __name__ == "__main__":
    main()