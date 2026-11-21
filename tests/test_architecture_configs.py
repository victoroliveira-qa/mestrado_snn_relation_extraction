from pathlib import Path
import yaml
from src.utils.config import load_config


def test_all_architectures_have_example_config():
    root=Path(__file__).resolve().parents[1]
    examples={'mlp','mlp_prototype','prototype','snn','snn_mlp','snn_prototype'}
    values=[]
    for file in (root/'configs'/'experiments').glob('business_*.yaml'):
        cfg=load_config(file)
        values.append(cfg['architecture']['name'])
    base=load_config(root/'configs'/'experiments'/'business_snn_prototype.yaml')
    values.append(base['architecture']['name'])
    assert set(values)==examples
