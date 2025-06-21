#!/usr/bin/env python3
# ingest_milvus.py

import glob
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from transformers import AutoTokenizer
import numpy as np
import requests
from embed_client import get_embedding
# ── CONFIG ─────────────────────────────────────────────────────────
TRITON_URL = "http://localhost:8002/v2/models/embedding_model/infer"
DOCS_DIR   = "/path/to/your/docs"     # e.g. txt files
COLL_NAME  = "text_embeddings"

# ── SETUP ──────────────────────────────────────────────────────────
# 1) Triton tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 2) Milvus connection + (re)create collection
connections.connect("default", host="127.0.0.1", port="19530")

# If collection already exists, skip these lines.
fields = [
    FieldSchema(name="id",        dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="source",    dtype=DataType.VARCHAR, max_length=256)
]
schema = CollectionSchema(fields, description="Doc chunk embeddings")
if not Collection.exists(COLL_NAME):
    Collection(COLL_NAME, schema)

coll = Collection(COLL_NAME)

# ── FUNCTIONS ──────────────────────────────────────────────────────
def get_embedding(text: str) -> np.ndarray:
    toks = tokenizer(text, padding="max_length", truncation=True,
                     max_length=16, return_tensors="np")
    payload = {"inputs": [
        {"name":"input_ids",      "shape": toks["input_ids"].shape.tolist(),
         "datatype":"INT64",      "data": toks["input_ids"].tolist()},
        {"name":"attention_mask", "shape": toks["attention_mask"].shape.tolist(),
         "datatype":"INT64",      "data": toks["attention_mask"].tolist()}
    ]}
    r = requests.post(TRITON_URL, json=payload)
    r.raise_for_status()
    vec = np.array(r.json()["outputs"][0]["data"][0], dtype=np.float32)
    return vec

# ── INGEST ─────────────────────────────────────────────────────────
ids, vecs, metas = [], [], []

for idx, path in enumerate(glob.glob(f"{DOCS_DIR}/*.txt"), start=1):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    emb = get_embedding(text)
    ids.append(idx)
    vecs.append(emb.tolist())
    metas.append(path.split("/")[-1])

# 4) Insert into Milvus
coll.insert([ids, vecs, metas])
coll.flush()

print(f"✅  Inserted {len(ids)} docs into `{COLL_NAME}`.")
