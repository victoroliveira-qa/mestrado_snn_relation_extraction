from pathlib import Path
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "configs" / "experiments"


def test_six_primary_architectures():
    expected = {"mlp", "mlp_prototype", "prototype", "snn", "snn_mlp", "snn_prototype"}
    files = sorted(EXP.glob("business_*.yaml"))
    assert len(files) == 6
    assert {load_config(p)["architecture"]["name"] for p in files} == expected


def test_nested_generated_embeddings_load():
    cfg = load_config(EXP / "other_embeddings" / "bge_m3_business_generated.yaml")
    assert cfg["embedding"]["model"] == "bge-m3"
    assert cfg["architecture"]["name"] == "snn_prototype"


def test_archived_experiment_loads():
    cfg = load_config(EXP / "archived" / "v3_qwen_business.yaml")
    assert cfg["architecture"]["name"] == "snn_prototype"


def test_unique_experiment_names():
    names = [load_config(p)["experiment"]["name"] for p in EXP.rglob("*.yaml")]
    assert len(names) == len(set(names))
