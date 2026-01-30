from langchain.schema import Document
from google.cloud import bigquery
from langchain.schema.retriever import BaseRetriever
from langchain_core.documents import Document
from typing import Any


class BigQueryVectorRetriever(BaseRetriever):
    client: Any
    table: str
    embeddings: Any
    strategy: str
    k: int

    def _get_relevant_documents(self, query: str):
        query_vec = self.embeddings.embed_query(query)

        sql = f"""
        SELECT content, title
        FROM `{self.table}`
        WHERE strategy = @strategy
        ORDER BY COSINE_DISTANCE(embedding_vector, @q)
        LIMIT {self.k}
        """

        job = self.client.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("q", "FLOAT64", query_vec),
                    bigquery.ScalarQueryParameter("strategy", "STRING", self.strategy),
                ]
            ),
        )

        return [
            Document(
                page_content=row.content,
                metadata={"title": row.title}
            )
            for row in job
        ]