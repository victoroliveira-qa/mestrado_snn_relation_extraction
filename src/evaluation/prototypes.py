import numpy as np


def compute_prototypes(projected, y, class_count):
    prototypes=[]
    missing=[]
    for label in range(class_count):
        vecs=projected[y==label]
        if len(vecs)==0:
            missing.append(label)
            prototypes.append(np.zeros(projected.shape[1],dtype=np.float32))
        else:
            v=vecs.mean(axis=0)
            prototypes.append(v/max(np.linalg.norm(v),1e-9))
    return np.asarray(prototypes,dtype=np.float32),missing


def classify(projected, prototypes, missing=()):
    proj=projected/np.maximum(np.linalg.norm(projected,axis=1,keepdims=True),1e-9)
    p=prototypes/np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-9)
    scores=proj @ p.T
    if len(missing):
        scores[:,list(missing)]=-np.inf
    return np.argmax(scores,axis=1),scores
