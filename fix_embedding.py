import pickle
from pathlib import Path

store_path = Path("backend-new/data/arcface_embeddings.pkl")
with store_path.open("rb") as f:
    data = pickle.load(f)

print("Before:", list(data.keys()))
data.pop("vivien", None)
print("After:", list(data.keys()))

with store_path.open("wb") as f:
    pickle.dump(data, f)