#!/usr/bin/env python3
# rag_milvus.py

import requests
from pymilvus import connections, Collection
import numpy as np
from embed_client import get_embedding   # reuse our Triton embed_client.py
import pickle

# ── CONFIG ─────────────────────────────────────────────────────────
COLL_NAME        = "text_embeddings"
HF_CHAT_ENDPOINT = "http://localhost:8001/v1/chat/completions"
CHAT_MODEL       = "qwen3-32b"
TOP_K            = 3

# ── SETUP ──────────────────────────────────────────────────────────
connections.connect("default", host="127.0.0.1", port="19530")
coll = Collection(COLL_NAME)

def retrieve(query: str, k=TOP_K):
    q_emb = get_embedding(query).tolist()
    search_params = {"metric_type":"L2", "params":{"nprobe":10}}
    results = coll.search(
        data=[q_emb],
        anns_field="embedding",
        param=search_params,
        limit=k,
        output_fields=["source"]
    )
    # results is List[List[Hit]]; we take the inner `.entity.get("source")`
    sources = [hit.entity.get("source") for hit in results[0]]
    return sources

def rag(query: str):
    docs = retrieve(query)
    # fetch the raw texts for context
    contexts = []
    for fn in docs:
        with open(f"/path/to/your/docs/{fn}", encoding="utf-8") as f:
            contexts.append(f"[{fn}]\n" + f.read())
    system_msgs = [
        {"role":"system", "content":"You are a helpful assistant. Use the context."},
        {"role":"system", "content": "\n\n---\n\n".join(contexts)}
    ]
    payload = {
        "model": CHAT_MODEL,
        "messages": system_msgs + [{"role":"user","content":query}]
    }
    r = requests.post(HF_CHAT_ENDPOINT, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    while True:
        q = input("\n👉 Question: ")
        print("\n📝 Answer:\n", rag(q))
