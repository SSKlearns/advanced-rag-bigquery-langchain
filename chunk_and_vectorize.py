import logging
import requests
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
import time
import os
import tarfile
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = "buildathon-485822"
REGION = "us-central1"
# Using the Wikipedia dataset
WIKIPEDIA_URL = "https://github.com/LGDoor/Dump-of-Simple-English-Wiki/raw/refs/heads/master/corpus.tgz"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MAX_DOCS_TO_PROCESS = 500 


def download_data():
    if os.path.exists("corpus.json"):
        logging.info("corpus.json already exists. Skipping download.")
        return

    logging.info(f"Downloading data from {WIKIPEDIA_URL}...")
    response = requests.get(WIKIPEDIA_URL)
    with open("corpus.tgz", "wb") as f:
        f.write(response.content)

    logging.info("Extracting tarball...")
    with tarfile.open("corpus.tgz", "r:gz") as tar:
        tar.extractall()

    logging.info("Converting to JSON...")
    output = []
    try:
        with open("corpus.txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n\n")
    except FileNotFoundError:
        logging.error("corpus.txt not found. Extraction might have failed.")
        return

    for block in lines:
        parts = block.split("\n", 1)
        if len(parts) != 2:
            continue
        title = parts[0].strip()
        text = parts[1].strip()
        if not text:
            continue
        output.append({"title": title, "text": text})

    with open("corpus.json", "w", encoding="utf-8") as out:
        json.dump(output, out)
    logging.info(f"Created corpus.json with {len(output)} articles.")

    return output


def get_splitter(strategy):
    if strategy == "character":
        return CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n")
    elif strategy == "token":
        return TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    else: # default to recursive
        return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def process_strategy(articles, strategy):
    splitter = get_splitter(strategy)
    rows = []
    
    logging.info(f"Processing {len(articles)} articles with strategy {strategy}...")

    # Process only a subset for the workshop
    for idx, article in enumerate(articles[:MAX_DOCS_TO_PROCESS]):
        # Handle different potential JSON structures
        # robustly check for multiple content keys
        text = article.get("text") or article.get("content") or article.get("page_content")
        title = article.get("title") or article.get("source") or f"doc_{idx}"
            
        if not text:
            continue
            
        try:
            chunks = splitter.split_text(text)
        except Exception as e:
            logging.warning(f"Failed to split doc {title}: {e}")
            continue
            
        for i, chunk in enumerate(chunks):
            rows.append({
                "doc_id": f"{idx}",
                "title": title,
                "chunk_id": i,
                "content": chunk,
                "strategy": strategy
            })
            
    return rows

def embed_chunks(rows, embeddings, batch_size=10, sleep_s=12):
    all_vectors = []
    
    total_batches = (len(rows) + batch_size - 1) // batch_size
    
    logging.info(f"Starting embedding generation for {len(rows)} chunks in {total_batches} batches...")

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        texts = [r["content"] for r in batch]

        try:
            try:
                vectors = embeddings.embed_documents(texts)
            except Exception as e:
                time.sleep(60)
                vectors = embeddings.embed_documents(texts)
            all_vectors.extend(vectors)
            
            if (i // batch_size) % 5 == 0:
                logging.info(f"Embedded batch {i//batch_size + 1}/{total_batches}")

            time.sleep(sleep_s) 
        except Exception as e:
            logging.error(f"Error embedding batch {i}: {e}")
            # If a batch fails, re-raise to stop execution or handle gracefully
            raise e

    # Assign embeddings back to rows
    for row, vec in zip(rows, all_vectors):
        row["embedding"] = vec

    return rows

def main():
    # 1. Load Data
    full_text = download_data()
    if not full_text:
        logging.error("No data found.")
        return

    # 2. Setup Config
    strategies = ["recursive", "character", "token"]
    
    # Check for GCP credentials (implicit check by instantiating class)
    try:
        embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=PROJECT_ID, location=REGION)
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI Embeddings. Check Auth: {e}")
        return

    # 3. Process Each Strategy
    for strategy in strategies:
        logging.info(f"--- Processing strategy: {strategy.upper()} ---")
        
        # Chunk
        rows = process_strategy(full_text, strategy)
        
        if not rows:
            logging.warning(f"No chunks generated for {strategy}.")
            continue
            
        # Embed
        logging.info(f"Generating embeddings for {len(rows)} chunks...")
        try:
            rows = embed_chunks(rows, embeddings)
        except Exception as e:
             logging.error(f"Skipping strategy {strategy} due to embedding error: {e}")
             continue

        # Save
        df = pd.DataFrame(rows)
        output_file = f"chunks_{strategy}.parquet"
        
        logging.info(f"Saving to {output_file}...")
        df.to_parquet(output_file, index=False)
        logging.info(f"Saved {output_file} successfully.")

if __name__ == "__main__":
    main()