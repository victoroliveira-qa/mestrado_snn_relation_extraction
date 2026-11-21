import numpy as np
from src.evaluation.prototypes import classify


def predict_embedding(embedding, encoder, prototypes, labels, missing=(), normalize_input=False, top_k=5):
    vec=np.asarray(embedding,dtype=np.float32).reshape(1,-1)
    if vec.shape[1]!=encoder.input_shape[-1]:
        raise ValueError(f"Embedding dimension {vec.shape[1]} != expected {encoder.input_shape[-1]}")
    if normalize_input:
        vec/=np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-9)
    projected=encoder.predict(vec,verbose=0)
    idx,scores=classify(projected,prototypes,missing)
    order=np.argsort(scores[0])[::-1]
    order=[i for i in order if np.isfinite(scores[0,i])][:top_k]
    return {"predicted_label":str(labels[idx[0]]),"top_k":[{"label":str(labels[i]),"cosine_similarity":float(scores[0,i])} for i in order]}
