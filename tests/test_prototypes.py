import numpy as np
from src.evaluation.prototypes import compute_prototypes, classify


def test_prototype_inference():
    x=np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]],dtype=np.float32)
    y=np.array([0,0,1,1])
    p,missing=compute_prototypes(x,y,2)
    pred,_=classify(x,p,missing)
    assert pred.tolist()==y.tolist()
