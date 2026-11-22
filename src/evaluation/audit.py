"""Reproducible experimental diagnostics. Reporting only: no effect on training."""
import numpy as np


def build_run_audit(history, y, split, class_names, embedding_audit):
    y = np.asarray(y)
    class_names = [str(x) for x in class_names]
    distributions = {}
    for name in ("train", "validation", "test"):
        ids = np.asarray(split[name], dtype=int)
        counts = np.bincount(y[ids], minlength=len(class_names))
        distributions[name] = {label: int(counts[i]) for i, label in enumerate(class_names)}
    raw = getattr(history, "history", {})
    curves = {str(key): [float(v) for v in values] for key, values in raw.items()}
    epoch_count = max((len(v) for v in curves.values()), default=0)
    losses = curves.get("loss", [])
    val_losses = curves.get("val_loss", [])
    valid_best = [(i, x) for i, x in enumerate(val_losses) if np.isfinite(x)]
    best = min(valid_best, key=lambda pair: pair[1]) if valid_best else None
    return {
        "samples": {key: int(sum(values.values())) for key, values in distributions.items()},
        "class_distribution": distributions,
        "training": {
            "epochs_executed": int(epoch_count),
            "last_epoch_loss": losses[-1] if losses else None,
            "last_epoch_validation_loss": val_losses[-1] if val_losses else None,
            "best_validation_loss": float(best[1]) if best else None,
            "best_validation_loss_epoch": int(best[0]) + 1 if best else None,
            "best_epoch_training_loss": losses[best[0]] if best and len(losses)>best[0] else None,
            "note": "Best epoch selected by minimum val_loss, not test performance. Final loss refers to last executed epoch."
        },
        "embedding_execution": dict(embedding_audit),
    }


def report_audit_lines(audit):
    if not audit:
        return []
    lines = ["", "DETALHAMENTO EXPERIMENTAL", "-" * 55]
    lines.append("AMOSTRAS: " + ", ".join(f"{key}={n}" for key, n in audit["samples"].items()))
    lines += ["", "DISTRIBUICAO DE CLASSES"]
    classes = list(audit["class_distribution"]["train"].keys())
    lines.append("Classe | Treino | Validacao | Teste")
    for cls in classes:
        dist = audit["class_distribution"]
        lines.append(f"{cls} | {dist['train'][cls]} | {dist['validation'][cls]} | {dist['test'][cls]}")
    train = audit["training"]
    lines += ["", "HISTORICO DE TREINAMENTO", f"Epocas executadas: {train['epochs_executed']}"]
    for title, key in [("Loss da ultima epoca", "last_epoch_loss"),
                       ("Validation loss da ultima epoca", "last_epoch_validation_loss"),
                       ("Melhor validation loss", "best_validation_loss"),
                       ("Epoca com menor validation loss", "best_validation_loss_epoch"),
                       ("Loss de treino nessa epoca", "best_epoch_training_loss")]:
        value = train[key]
        lines.append(f"{title}: {value if value is not None else 'NAO DISPONIVEL'}")
    embedding = audit["embedding_execution"]
    lines += ["", "EMBEDDINGS — PROVENIENCIA"]
    lines.append(f"Provedor: {embedding.get('provider', 'unknown')}")
    lines.append(f"Modelo: {embedding.get('model_name') or embedding.get('model') or 'NAO INFORMADO'}")
    lines.append(f"Dispositivo de geracao: {embedding.get('generation_device', 'NAO REGISTRADO')}")
    lines.append(f"Cache reutilizado: {embedding.get('cache_hit', 'NAO APLICAVEL')}")
    lines.append(f"Arquivo de origem: {embedding.get('source_file', 'NAO REGISTRADO')}")
    return lines


def report_class_lines(metrics):
    lines = []
    for split, payload in metrics.items():
        report = payload.get("classification_report") or payload.get("pair_report")
        if not isinstance(report, dict):
            continue
        lines += ["", f"PRECISION / RECALL / F1 POR CLASSE — {split.upper()}", "-" * 55,
                  "Classe | Precision | Recall | F1 | Suporte"]
        for name, data in report.items():
            if not isinstance(data, dict) or "f1-score" not in data or name in ("macro avg", "weighted avg", "micro avg", "samples avg"):
                continue
            lines.append(f"{name} | {data['precision']:.4f} | {data['recall']:.4f} | {data['f1-score']:.4f} | {int(data.get('support', 0))}")
    return lines
