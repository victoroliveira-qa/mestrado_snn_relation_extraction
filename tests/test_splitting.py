import numpy as np
from src.data.splitting import split_ids


def test_disjoint_splits():
    y=np.repeat(np.arange(3),40)
    result=split_ids(y,{"train":.8,"validation":.1,"test":.1,"stratify":"strict"},42)
    a,b,c=(set(result[k]) for k in ("train","validation","test"))
    assert not (a&b or b&c or a&c)
    assert len(a|b|c)==len(y)
