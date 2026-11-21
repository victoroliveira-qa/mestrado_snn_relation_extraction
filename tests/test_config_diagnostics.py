from pathlib import Path
import pytest
import yaml
from src.utils.config import load_config
from src.utils.runs import create_run, write_txt_report

ROOT = Path(__file__).resolve().parents[1]


def test_business_mlp_resolves_all_sections():
    cfg = load_config(ROOT / "configs/experiments/business_mlp.yaml")
    assert cfg["dataset"]["label_column"] == "relation_label"
    assert cfg["embedding"]["provider"] == "faiss"
    assert cfg["architecture"]["name"] == "mlp"


def test_incomplete_config_has_explicit_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump({"components": {"dataset":"business"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="components must define exactly"):
        load_config(path)
    path.write_text(yaml.safe_dump({"experiment": {"name": "bad", "seed": 42}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete experiment configuration"):
        load_config(path)


def test_error_report_accepts_missing_dataset(tmp_path):
    run, dirs = create_run(tmp_path, "failure")
    report = write_txt_report(dirs, {"experiment": {"seed": 42}, "components": {"dataset": "business"}}, {}, run, status="FALHOU", extra=["Erro simulado"])
    assert "NAO RESOLVIDO" in report.read_text(encoding="utf-8")
    assert "Erro simulado" in report.read_text(encoding="utf-8")
