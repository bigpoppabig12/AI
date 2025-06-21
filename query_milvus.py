# /home/rag/query_milvus.py
from pymilvus import connections, Collection
from embed_client import get_embedding

# connect
connections.connect(alias="default", host="127.0.0.1", port="19530")
coll = Collection("text_embeddings")

# your query
query_text = "Advances in AI and machine learning."
query_emb = get_embedding(query_text).tolist()

# search parameters
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

# perform search
results = coll.search(
    data=query_emb,
    anns_field="embedding",
    param=search_params,
    limit=3,
    expr=None,
    output_fields=["id"]
)

for hits in results:
    print(f"\nQuery: “{query_text}”")
    for hit in hits:
        print(f"  • id={hit.id}, distance={hit.distance:.4f}")
