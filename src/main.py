import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "candidate_job_data.csv")
CHROMA_PERSIST_DIR = os.path.join(SCRIPT_DIR, "..", "chroma_job_db")

# Globals
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
candidates_collection = chroma_client.get_or_create_collection("candidates")
jobs_collection = chroma_client.get_or_create_collection("jobs")

def load_and_embed():
    df = pd.read_csv(DATA_PATH)
    candidates = df[df["type"] == "candidate"]
    jobs = df[df["type"] == "job"]

    if candidates_collection.count() == 0:
        candidate_embeddings = embedding_model.encode(candidates["text"].tolist(), convert_to_numpy=True).tolist()
        metadatas = candidates[["name_or_title", "text"]].rename(columns={"name_or_title": "name", "text": "resume"}).to_dict(orient="records")
        candidates_collection.add(ids=candidates["id"].astype(str).tolist(),
                                  documents=candidates["text"].tolist(),
                                  embeddings=candidate_embeddings,
                                  metadatas=metadatas)

    if jobs_collection.count() == 0:
        job_embeddings = embedding_model.encode(jobs["text"].tolist(), convert_to_numpy=True).tolist()
        metadatas = jobs[["name_or_title", "text"]].rename(columns={"name_or_title": "title", "text": "description"}).to_dict(orient="records")
        jobs_collection.add(ids=jobs["id"].astype(str).tolist(),
                            documents=jobs["text"].tolist(),
                            embeddings=job_embeddings,
                            metadatas=metadatas)

def find_candidates_for_job(job_desc, top_k=3):
    if chroma_client is None or candidates_collection.count() == 0:
        load_and_embed()

    q_embedding = embedding_model.encode([job_desc], convert_to_numpy=True).tolist()
    results = candidates_collection.query(query_embeddings=q_embedding,
                                          n_results=top_k,
                                          include=["metadatas", "distances"])
    return results

if __name__ == "__main__":
    load_and_embed()
    job_description = "Looking for a Python developer with Django experience"
    matches = find_candidates_for_job(job_description)
    for meta, dist in zip(matches["metadatas"][0], matches["distances"][0]):
        print(f"Candidate: {meta['name']}, Similarity Score: {1 - dist:.2f}")
        print(f"Resume Summary: {meta['resume'][:100]}...\n")
