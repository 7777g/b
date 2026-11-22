# Run once while building: downloads the small embedding model so the server does not have to download it later.
import os
from fastembed import TextEmbedding

HERE = os.path.dirname(os.path.abspath(__file__))
model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=os.path.join(HERE, "models"), threads=1)
print("model ready, size:", len(next(iter(model.embed(["hello"])))))
