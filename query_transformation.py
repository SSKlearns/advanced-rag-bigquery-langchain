import os
import logging
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_core.prompts import PromptTemplate
from BigQueryRetriever import BigQueryVectorRetriever
from google.cloud import bigquery

logging.basicConfig(level=logging.ERROR)

PROJECT_ID = "buildathon-485822"
REGION = "us-central1"
BQ_DATASET = "advanced_rag"
BQ_TABLE = "wikipedia_vectors"


def generate_hyde_doc(query, llm):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Write a concise, hypothetical answer to the question. Question: {question} Answer:"
    )
    chain = prompt | llm
    return chain.invoke({"question": query})


def generate_step_back(query, llm):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Write a more general, abstract question that concepts in this question. Original: {question} Step-back:"
    )
    chain = prompt | llm
    return chain.invoke({"question": query})


def main():
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)
    llm = VertexAI(model_name="gemini-2.5-flash", project=PROJECT_ID, location=REGION, temperature=0.5)
    bq_client = bigquery.Client(project=PROJECT_ID)

    bq_retriever = BigQueryVectorRetriever(
                    client=bq_client,
                    table="buildathon-485822.advanced_rag.wikipedia_vectors",
                    embeddings=embeddings,
                    strategy="recursive",
                    k=20
                )

    original_query = "Tell me about the Dursleys."
    print(f"ORIGINAL QUERY: {original_query}\n" + "-"*30)

    # 1. HyDE
    hyde_doc = generate_hyde_doc(original_query, llm)
    print(f"HyDE Generated Doc: {hyde_doc.strip()[:100]}...")
    hyde_results = bq_retriever.invoke(hyde_doc)
    print(f"HyDE Retrieval: {hyde_results[0].page_content[:100]}...\n")

    # 2. Step-back
    step_back_q = generate_step_back(original_query, llm)
    print(f"Step-back Query: {step_back_q.strip()}")
    step_results = bq_retriever.invoke(step_back_q)
    print(f"Step-back Retrieval: {step_results[0].page_content[:100]}...")

if __name__ == "__main__":
    main()