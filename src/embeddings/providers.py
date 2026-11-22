from pathlib import Path
import numpy as np


def load_vectors(config, df, root, audit=None):
    """Load already-computed embeddings without changing their row order."""
    from src.utils.config import resolve_path
    provider = config["provider"].lower()
    if audit is not None:
        audit.update({"provider": provider, "model": config.get("model"),
                      "generation_device": "not_recorded", "cache_hit": None})
    if provider in ("model", "sentence_transformers"):
        from src.embeddings.model_registry import generate_vectors
        return generate_vectors(config, df, root, audit=audit)
    path = resolve_path(config["path"], root)
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings not found: {path}")
    if provider == "npy":
        all_vectors = np.load(path, mmap_mode="r")
        if all_vectors.ndim != 2 or len(all_vectors) != len(df):
            raise ValueError("NPY must be a matrix with exactly one vector per original dataset row")
        vectors = np.asarray(all_vectors, dtype=np.float32)
    elif provider == "faiss":
        if not config.get("positional_alignment_verified", False):
            raise ValueError("FAISS positional alignment is unverified. Audit source ordering, then explicitly set positional_alignment_verified: true")
        import faiss
        index = faiss.read_index(str(path))
        if index.ntotal != len(df):
            raise ValueError(f"FAISS has {index.ntotal} vectors but dataset has {len(df)} rows. Positional alignment cannot be assumed.")
        try:
            vectors = np.vstack([index.reconstruct(int(i)) for i in df["_original_row_id"]]).astype(np.float32)
        except Exception as exc:
            raise RuntimeError("FAISS index does not support required reconstruction; export exact vectors or use NPY provider") from exc
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Supported: faiss, npy")
    if audit is not None:
        audit["generation_device"] = "unknown (precomputed; original generation environment not recorded)"
        audit["source_file"] = str(path)
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite embedding values detected")
    if config.get("normalize_input", False):
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    return vectors
