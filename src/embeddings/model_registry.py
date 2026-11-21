"""Model-based embedding generation with explicit input construction and cache validation."""
import hashlib
import json
from pathlib import Path
import numpy as np

MODEL_REGISTRY = {
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
    "bge-m3": "BAAI/bge-m3",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
}


def build_texts(df, config):
    columns = config.get("text_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("embedding.text_columns must list source columns for model embeddings")
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns for embeddings: {missing}")
    if df[columns].isna().any().any():
        raise ValueError("Text columns contain nulls; clean them explicitly before generating embeddings")
    separator = config.get("separator", "\n")
    texts = [separator.join(f"{name}: {row[name]}" for name in columns) for _, row in df[columns].iterrows()]
    prefix = config.get("text_prefix", "")
    return [prefix + text for text in texts]


def generate_vectors(config, df, root):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install optional dependencies: pip install -r requirements-embeddings.txt") from exc
    model_key = config.get("model")
    model_name = config.get("model_name") or MODEL_REGISTRY.get(model_key)
    if not model_name:
        raise ValueError(f"Unknown model '{model_key}'. Set embedding.model_name to a Hugging Face model ID")
    texts = build_texts(df, config)
    config_identity = {
        "model": model_name,
        "revision": config.get("revision"),
        "text_columns": config["text_columns"],
        "separator": config.get("separator", "\n"),
        "text_prefix": config.get("text_prefix", ""),
        "prompt_name": config.get("prompt_name"),
        "prompt": config.get("prompt"),
        "normalize_input": bool(config.get("normalize_input", False)),
        "truncate_dim": config.get("truncate_dim"),
        "texts_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(config_identity, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(root) / 'datasets' / 'embeddings' / 'generated'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (fingerprint + '.npy')
    manifest = cache_dir / (fingerprint + '.json')
    if cache.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == fingerprint and saved.get("identity") == config_identity:
            arr = np.load(cache, allow_pickle=False)
            if arr.ndim == 2 and len(arr) == len(df) and np.isfinite(arr).all():
                print(f"Reusing verified embedding cache: {cache}")
                return np.asarray(arr, dtype=np.float32)
        raise ValueError("Embedding cache has inconsistent metadata or shape")
    print(f"Generating embeddings with {model_name}; this may require substantial GPU memory")
    args = {"trust_remote_code": bool(config.get("trust_remote_code", False))}
    if config.get("revision"):
        args["revision"] = config["revision"]
    if config.get("truncate_dim"):
        args["truncate_dim"] = int(config["truncate_dim"])
    model = SentenceTransformer(model_name, **args)
    kwargs = {"batch_size": int(config.get("batch_size", 8)), "show_progress_bar": True, "convert_to_numpy": True,
              "normalize_embeddings": bool(config.get("normalize_input", False))}
    if config.get("prompt_name"):
        kwargs["prompt_name"] = config["prompt_name"]
    elif config.get("prompt"):
        kwargs["prompt"] = config["prompt"]
    array = np.asarray(model.encode(texts, **kwargs), dtype=np.float32)
    if array.ndim != 2 or len(array) != len(df) or not np.isfinite(array).all():
        raise ValueError("Invalid generated embeddings")
    np.save(cache, array)
    manifest.write_text(json.dumps({"fingerprint": fingerprint, "identity": config_identity,
                                    "shape": list(array.shape)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Cached embeddings: {cache}")
    return array
