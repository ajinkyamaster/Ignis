from functools import lru_cache
import os

from app.config import settings

BGE_REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'

@lru_cache(maxsize=1)
def model():
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_dir = os.getenv('EMBEDDING_MODEL_DIR') or snapshot_download(
        settings.embedding_model,
        revision=BGE_REVISION if settings.embedding_model == 'BAAI/bge-small-en-v1.5' else None,
        allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                        'special_tokens_map.json', 'vocab.txt', 'onnx/model.onnx'],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        os.path.join(model_dir, 'onnx', 'model.onnx'),
        sess_options=options,
        providers=['CPUExecutionProvider'],
    )
    return tokenizer, session

def encode(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    import numpy as np

    tokenizer, session = model()
    tokens = tokenizer(texts, padding=True, truncation=True,
                       max_length=tokenizer.model_max_length, return_tensors='np')
    inputs = {item.name: tokens[item.name] for item in session.get_inputs()}
    # BGE-small's SentenceTransformer configuration pools the CLS token, then
    # L2-normalizes it. This preserves the vectors already stored in Postgres.
    vectors = session.run(None, inputs)[0][:, 0, :]
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors.tolist()

def cosine(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b))
