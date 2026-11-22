import numpy as np
from src.evaluation.audit import build_run_audit, report_audit_lines, report_class_lines


def test_audit_counts_and_epoch():
    class History:
        history = {"loss": [0.9, 0.6, 0.5], "val_loss": [0.8, 0.4, 0.7]}
    y = np.array([0, 1, 0, 1, 0, 1])
    split = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
    audit = build_run_audit(History(), y, split, ["A", "B"], {"generation_device": "cuda:0"})
    assert audit["samples"] == {"train": 2, "validation": 2, "test": 2}
    assert audit["training"]["best_validation_loss_epoch"] == 2
    assert audit["training"]["last_epoch_loss"] == 0.5
    assert audit["class_distribution"]["test"] == {"A": 1, "B": 1}
    assert "Dispositivo de geracao: cuda:0" in report_audit_lines(audit)


def test_per_class_reporting():
    payload = {"test": {"classification_report": {"A": {"precision": 0.8, "recall": 0.5, "f1-score": 0.615, "support": 4}, "macro avg": {"precision": 0.8, "recall": 0.5, "f1-score": 0.615}}}}
    assert any("A | 0.8000 | 0.5000 | 0.6150 | 4" == line for line in report_class_lines(payload))
