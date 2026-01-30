# advanced-rag-bigquery-langchain

This repository contains the setup steps for running **Advanced Retrieval-Augmented Generation (RAG)** pipelines using **Google Cloud Platform (GCP)** and **BigQuery** as the vector store.

The goal is to:

* Load pre-computed embeddings (Parquet)
* Store them in BigQuery
* Use BigQuery for vector similarity search and advanced RAG techniques

---

## Prerequisites

Before starting, make sure you have:

* A **GCP project** with billing enabled
* **Google Cloud SDK** installed
  👉 [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
* Authenticated with GCP:

  ```bash
  gcloud auth login
  gcloud config set project YOUR_PROJECT_ID
  ```
* `bq` CLI available:

  ```bash
  bq --version
  ```

---

## 1️⃣ Enable Required GCP Services

Run the following command to enable all required APIs:

```bash
gcloud services enable \
  dataplex.googleapis.com \
  dataform.googleapis.com \
  storage-component.googleapis.com \
  aiplatform.googleapis.com \
  datalineage.googleapis.com \
  artifactregistry.googleapis.com \
  visionai.googleapis.com \
  cloudapiregistry.googleapis.com \
  telemetry.googleapis.com \
  notebooks.googleapis.com \
  dataflow.googleapis.com \
  compute.googleapis.com \
  discoveryengine.googleapis.com
```

⏳ This may take a few minutes the first time.

---

## 2️⃣ Create a BigQuery Dataset

Create a BigQuery dataset (replace the name if needed):

```bash
bq mk --location=us-central1 advanced_rag
```

This dataset will store:

* Chunked text
* Embeddings
* Vector-ready tables for retrieval

---

## 3️⃣ Upload Parquet Files to BigQuery

Assuming your Parquet files are already uploaded to **Google Cloud Storage**:

```bash
gs://advanced-rag-bucket/*.parquet
```

Load them into BigQuery:

```bash
bq load \
  --source_format=PARQUET \
  advanced_rag.wikipedia_chunks \
  gs://advanced-rag-bucket/*.parquet
```

This creates a table:

```
advanced_rag.wikipedia_chunks
```

---

## 4️⃣ Flatten Embeddings for Vector Search

If your embeddings are stored as nested structures (e.g. `embedding.list`), BigQuery vector functions require a **flat ARRAY<FLOAT64>**.

Run the following query in the **BigQuery SQL Editor**:

```sql
CREATE OR REPLACE TABLE advanced_rag.wikipedia_vectors AS
SELECT
  *,
  ARRAY(
    SELECT element
    FROM UNNEST(embedding.list)
  ) AS embedding_vector
FROM advanced_rag.wikipedia_chunks;
```

This creates a vector-ready table:

```
advanced_rag.wikipedia_vectors
```

Where:

* `embedding_vector` is `ARRAY<FLOAT64>`
* Compatible with `COSINE_DISTANCE`, `DOT_PRODUCT`, etc.

---

## 5️⃣ What You Have Now

At this point, you have:

* ✅ Chunked documents
* ✅ Pre-computed embeddings
* ✅ Stored in BigQuery
* ✅ Ready for:

  * Vector similarity search
  * Hybrid RAG
  * Re-ranking
  * Contextual compression
  * Advanced RAG experiments

---

## Why BigQuery for RAG?

* No database setup or scaling concerns
* SQL-native vector operations
* Easy integration with Vertex AI
* Ideal for workshops and demos
* Handles tens of thousands of vectors effortlessly

---

## Next Steps

You can now:

* Run cosine similarity search in SQL
* Plug BigQuery retrieval into a LangChain / custom retriever
* Demonstrate re-ranking and contextual compression
* Compare chunking strategies and retrieval quality
