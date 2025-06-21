# /home/rag/embed_client.py
import requests
from transformers import AutoTokenizer
import numpy as np

TRITON_URL = "http://localhost:8002/v2/models/embedding_model/infer"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    # tokenize to fixed length
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="np"
    )
    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": inputs["input_ids"].shape.tolist(),
                "datatype": "INT64",
                "data": inputs["input_ids"].tolist()
            },
            {
                "name": "attention_mask",
                "shape": inputs["attention_mask"].shape.tolist(),
                "datatype": "INT64",
                "data": inputs["attention_mask"].tolist()
            }
        ]
    }
    resp = requests.post(TRITON_URL, json=payload)
    resp.raise_for_status()
    out = resp.json()["outputs"][0]["data"]
    return np.array(out, dtype=np.float32)

if __name__ == "__main__":
    emb = get_embedding("This is a test sentence.")
    print("Shape:", emb.shape)  # should be (1,384)
