"""
SNN Research — Consolidador de Resultados Experimentais

Gera 7 arquivos CSV a partir das execuções armazenadas em results/.

Uso:
    python -m src.evaluation.consolidate_results \
        --input results \
        --output results/consolidated

Dependências:
    pip install pandas pyyaml
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ARCHITECTURES = (
    "snn_prototype",
    "mlp_prototype",
    "snn_mlp",
    "prototype",
    "snn",
    "mlp",
)

EMBEDDING_ALIASES = {
    "qwen8b_faiss": "Qwen3-Embedding-8B (FAISS)",
    "qwen8b": "Qwen3-Embedding-8B",
    "qwen4b": "Qwen3-Embedding-4B",
    "qwen3-embedding-8b": "Qwen3-Embedding-8B",
    "qwen3-embedding-4b": "Qwen3-Embedding-4B",
    "bge_m3": "BGE-M3",
    "bge-m3": "BGE-M3",
    "multilingual_e5": "Multilingual-E5-Large",
    "multilingual-e5-large": "Multilingual-E5-Large",
    "text-embedding-3-large": "OpenAI text-embedding-3-large",
    "openai_3_large": "OpenAI text-embedding-3-large",
}

CSV_FILES = {
    "01_experimentos.csv": "experiments",
    "02_metricas_por_classe.csv": "classes",
    "03_treinamento.csv": "training",
    "04_comparativo_interno.csv": "internal",
    "05_baselines_cientificos.csv": "baselines",
    "06_comparacao_externa.csv": "external",
    "07_dashboard_dados.csv": "dashboard",
}

EXPERIMENT_COLUMNS = [
    "run_id",
    "experiment",
    "dataset",
    "embedding",
    "embedding_provider",
    "embedding_device",
    "architecture",
    "status",
    "seed",
    "protocol",
    "train_size",
    "validation_size",
    "test_size",
    "accuracy_validation",
    "macro_f1_validation",
    "precision_macro_validation",
    "recall_macro_validation",
    "accuracy_test",
    "macro_f1_test",
    "precision_macro_test",
    "recall_macro_test",
    "run_path",
]

CLASS_COLUMNS = [
    "run_id",
    "dataset",
    "embedding",
    "architecture",
    "split",
    "class",
    "precision",
    "recall",
    "f1",
    "support",
]

TRAINING_COLUMNS = [
    "run_id",
    "dataset",
    "embedding",
    "architecture",
    "epochs_executed",
    "final_loss",
    "final_validation_loss",
    "best_validation_loss",
    "best_epoch",
]

BASELINE_COLUMNS = [
    "article",
    "year",
    "dataset",
    "protocol",
    "method",
    "metric",
    "value",
    "source",
    "table_or_page",
    "verified",
]

EXTERNAL_COLUMNS = [
    "article",
    "dataset",
    "protocol",
    "baseline_method",
    "architecture",
    "embedding",
    "metric",
    "our_value",
    "published_value",
    "difference",
    "source",
    "comparison_status",
]

METRIC_ALIASES = {
    "accuracy": ("accuracy", "acc"),
    "macro_f1": ("macro_f1", "f1_macro", "macro-f1"),
    "precision_macro": (
        "precision_macro",
        "macro_precision",
    ),
    "recall_macro": (
        "recall_macro",
        "macro_recall",
    ),
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}

    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)

        return value if isinstance(value, dict) else {}

    except (OSError, UnicodeError, ValueError) as exc:
        warnings.warn(f"Erro lendo {path}: {exc}")
        return {}


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}

    try:
        with path.open(encoding="utf-8") as file:
            value = yaml.safe_load(file)

        return value if isinstance(value, dict) else {}

    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        warnings.warn(f"Erro lendo {path}: {exc}")
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except (OSError, ValueError, UnicodeError) as exc:
        warnings.warn(f"Erro lendo {path}: {exc}")
        return pd.DataFrame()


def get_nested(data: Any, *keys, default=None):
    value = data

    for key in keys:
        if not isinstance(value, dict):
            return default

        value = value.get(key)

    return default if value is None else value


def first_value(*values):
    for value in values:
        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, (dict, list)) and not value:
            continue

        return value

    return None


def as_number(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", "."))
        except ValueError:
            return None

    return None


def normalize_dataset(value):
    if value is None:
        return None

    name = str(value).lower()

    if "core" in name:
        return "CORE"

    if "business" in name or "dataset_treino_base" in name:
        return "Business"

    return str(value)


def normalize_embedding(value):
    if value is None:
        return None

    if isinstance(value, dict):
        value = first_value(
            value.get("model"),
            value.get("name"),
            value.get("provider"),
        )

    if value is None:
        return None

    name = str(value).strip()
    lowered = name.lower()

    if lowered in EMBEDDING_ALIASES:
        return EMBEDDING_ALIASES[lowered]

    # Modelos informados como identificadores Hugging Face.
    for alias, title in sorted(
        EMBEDDING_ALIASES.items(),
        key=lambda pair: len(pair[0]),
        reverse=True,
    ):
        if alias in lowered:
            return title

    return name


def detect_architecture(*values):
    for value in values:
        if value is None:
            continue

        name = str(value).lower().replace("-", "_")

        for architecture in ARCHITECTURES:
            if (
                name == architecture
                or name.endswith("_" + architecture)
                or f"/{architecture}/" in name
            ):
                return architecture

    return None


def read_report(run: Path):
    report_dir = run / "reports"

    if not report_dir.is_dir():
        return {}, {}

    candidates = sorted(
        report_dir.glob("*.txt"),
        key=lambda path: path.stat().st_mtime,
    )

    if not candidates:
        return {}, {}

    text = candidates[-1].read_text(
        encoding="utf-8",
        errors="replace",
    )

    def field(name):
        match = re.search(
            rf"(?im)^\s*{re.escape(name)}:\s*(.+?)\s*$",
            text,
        )
        return match.group(1).strip() if match else None

    embedding_raw = field("Embedding")
    embedding_value = embedding_raw

    if embedding_raw and embedding_raw.startswith("{"):
        try:
            embedding_value = json.loads(embedding_raw)
        except ValueError:
            pass

    fields = {
        "status": field("Status"),
        "architecture": field("Arquitetura"),
        "dataset": field("Dataset"),
        "embedding": embedding_value,
        "seed": field("Seed"),
    }

    # Recuperação de métricas quando o JSON não está disponível.
    report_metrics = {"validation": {}, "test": {}}
    current_split = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.upper() == "[VALIDATION]":
            current_split = "validation"
            continue

        if stripped.upper() == "[TEST]":
            current_split = "test"
            continue

        if current_split is None:
            continue

        match = re.search(
            r"(?i)(accuracy|macro_f1|f1_macro)"
            r"\s*:\s*([0-9]+(?:[.,][0-9]+)?)",
            stripped,
        )

        if match:
            key = match.group(1).lower()
            report_metrics[current_split][key] = as_number(
                match.group(2)
            )

    return fields, report_metrics


def load_run_config(run: Path):
    config_dir = run / "configs"

    if not config_dir.is_dir():
        return {}

    preferred = [
        "resolved_config.yaml",
        "config.yaml",
        "experiment.yaml",
        "resolved.yaml",
    ]

    for filename in preferred:
        file = config_dir / filename

        if file.is_file():
            return read_yaml(file)

    files = sorted(
        list(config_dir.glob("*.yaml"))
        + list(config_dir.glob("*.yml"))
    )

    # Evita selecionar um componente isolado como se
    # fosse a configuração efetiva da execução.
    for file in files:
        data = read_yaml(file)

        if (
            "dataset" in data
            and "embedding" in data
            and "architecture" in data
        ):
            return data

    return {}


def read_run_metadata(run: Path):
    files = [
        run / "metadata.json",
        run / "logs" / "metadata.json",
        run / "metrics" / "metadata.json",
    ]

    for file in files:
        if file.is_file():
            return read_json(file)

    return {}


def find_metric_file(run: Path, split: str):
    names = {
        "test": [
            "test_metrics.json",
            "metrics_test.json",
        ],
        "validation": [
            "validation_metrics.json",
            "val_metrics.json",
            "metrics_validation.json",
        ],
    }

    for folder in (run / "metrics", run):
        for name in names[split]:
            path = folder / name

            if path.is_file():
                return read_json(path)

    return {}


def metric_container(data: dict, split: str):
    if not isinstance(data, dict):
        return {}

    candidates = [
        data,
        data.get("metrics"),
        data.get(split),
        get_nested(data, "metrics", split),
    ]

    for item in candidates:
        if not isinstance(item, dict):
            continue

        if any(
            key in item
            for aliases in METRIC_ALIASES.values()
            for key in aliases
        ):
            return item

    return data


def extract_metric(data: dict, name: str):
    if not isinstance(data, dict):
        return None

    for alias in METRIC_ALIASES[name]:
        value = as_number(data.get(alias))

        if value is not None:
            return value

    class_report = first_value(
        data.get("classification_report"),
        data.get("per_class"),
        data.get("class_report"),
        {},
    )

    macro = class_report.get("macro avg", {})

    if name == "macro_f1":
        return as_number(
            first_value(
                macro.get("f1-score"),
                macro.get("f1"),
            )
        )

    if name == "precision_macro":
        return as_number(macro.get("precision"))

    if name == "recall_macro":
        return as_number(macro.get("recall"))

    return None


def extract_class_metrics(
    data: dict,
    run_id: str,
    dataset,
    embedding,
    architecture,
    split: str,
):
    container = metric_container(data, split)

    report = first_value(
        container.get("classification_report"),
        container.get("per_class"),
        container.get("class_report"),
        {},
    )

    if not isinstance(report, dict):
        return []

    ignored = {
        "accuracy",
        "macro avg",
        "weighted avg",
        "micro avg",
        "samples avg",
    }

    records = []

    for label, metrics in report.items():
        if label in ignored:
            continue

        if not isinstance(metrics, dict):
            continue

        records.append({
            "run_id": run_id,
            "dataset": dataset,
            "embedding": embedding,
            "architecture": architecture,
            "split": split,
            "class": label,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": first_value(
                metrics.get("f1-score"),
                metrics.get("f1"),
            ),
            "support": metrics.get("support"),
        })

    return records


def load_training_history(run: Path):
    paths = [
        run / "metrics" / "training_history.csv",
        run / "logs" / "training_history.csv",
        run / "training_history.csv",
        run / "metrics" / "history.csv",
    ]

    for path in paths:
        if path.is_file():
            return read_csv(path)

    return pd.DataFrame()


def extract_training(
    run_id,
    dataset,
    embedding,
    architecture,
    history: pd.DataFrame,
):
    result = {
        "run_id": run_id,
        "dataset": dataset,
        "embedding": embedding,
        "architecture": architecture,
        "epochs_executed": None,
        "final_loss": None,
        "final_validation_loss": None,
        "best_validation_loss": None,
        "best_epoch": None,
    }

    if history.empty:
        return result

    result["epochs_executed"] = len(history)

    def column_values(names):
        for name in names:
            if name in history.columns:
                values = pd.to_numeric(
                    history[name],
                    errors="coerce",
                )
                return values, name

        return None, None

    loss, _ = column_values(
        ("loss", "train_loss", "training_loss")
    )

    if loss is not None:
        available = loss.dropna()

        if not available.empty:
            result["final_loss"] = float(available.iloc[-1])

    validation_loss, _ = column_values(
        ("val_loss", "validation_loss", "valid_loss")
    )

    if validation_loss is not None:
        available = validation_loss.dropna()

        if not available.empty:
            best_index = available.idxmin()

            result["final_validation_loss"] = float(
                available.iloc[-1]
            )

            result["best_validation_loss"] = float(
                available.loc[best_index]
            )

            if "epoch" in history.columns:
                result["best_epoch"] = history.loc[
                    best_index, "epoch"
                ]
            else:
                result["best_epoch"] = (
                    history.index.get_loc(best_index) + 1
                )

    return result


def discover_runs(root: Path, output: Path):
    found = set()
    output = output.resolve()

    def inside_output(path):
        try:
            path.resolve().relative_to(output)
            return True
        except ValueError:
            return False

    for name in (
        "test_metrics.json",
        "validation_metrics.json",
        "metadata.json",
    ):
        for file in root.rglob(name):
            if inside_output(file):
                continue

            parent = file.parent

            if parent.name in ("metrics", "logs"):
                run = parent.parent
            else:
                run = parent

            if (
                (run / "metrics").is_dir()
                or (run / "reports").is_dir()
            ):
                found.add(run)

    for file in root.rglob("relatorio_*.txt"):
        if inside_output(file):
            continue

        if file.parent.name == "reports":
            found.add(file.parent.parent)

    return sorted(found)


def collect_run(run: Path, root: Path):
    relative = run.relative_to(root)
    experiment = relative.parts[0]

    # Usa o caminho relativo inteiro como identificador.
    run_id = relative.as_posix()

    metadata = read_run_metadata(run)
    config = load_run_config(run)
    report, report_metrics = read_report(run)

    audit = read_json(
        run / "logs" / "experimental_audit.json"
    )

    dataset_config = config.get("dataset", {})
    embedding_config = config.get("embedding", {})
    architecture_config = config.get("architecture", {})

    if not isinstance(dataset_config, dict):
        dataset_config = {"name": dataset_config}

    if not isinstance(embedding_config, dict):
        embedding_config = {"model": embedding_config}

    if not isinstance(architecture_config, dict):
        architecture_config = {
            "name": architecture_config
        }

    dataset = normalize_dataset(
        first_value(
            get_nested(metadata, "dataset", "path"),
            metadata.get("dataset"),
            dataset_config.get("path"),
            dataset_config.get("name"),
            report.get("dataset"),
            experiment,
        )
    )

    architecture = detect_architecture(
        report.get("architecture"),
        metadata.get("architecture"),
        architecture_config.get("name"),
        architecture_config.get("type"),
        experiment,
    )

    embedding_info = first_value(
        metadata.get("embedding"),
        embedding_config,
        report.get("embedding"),
    )

    if isinstance(embedding_info, dict):
        embedding_raw = first_value(
            embedding_info.get("model"),
            embedding_info.get("name"),
            report.get("embedding"),
        )
        provider = embedding_info.get("provider")
    else:
        embedding_raw = embedding_info
        provider = None

    embedding = normalize_embedding(embedding_raw)

    # Diferencia Qwen FAISS de geração pelo modelo.
    if (
        embedding
        and "Qwen3-Embedding-8B" in embedding
        and (
            provider == "faiss"
            or "faiss" in str(
                first_value(
                    embedding_config.get("provider"),
                    embedding_info,
                    "",
                )
            ).lower()
            or "qwenfaiss" in experiment.lower()
        )
    ):
        embedding = "Qwen3-Embedding-8B (FAISS)"

    test_data = find_metric_file(run, "test")
    validation_data = find_metric_file(run, "validation")

    test = metric_container(test_data, "test")
    validation = metric_container(
        validation_data, "validation"
    )

    # Relatório TXT é a fonte alternativa.
    for key, value in report_metrics["test"].items():
        test.setdefault(key, value)

    for key, value in report_metrics["validation"].items():
        validation.setdefault(key, value)

    row = {
        "run_id": run_id,
        "experiment": experiment,
        "dataset": dataset,
        "embedding": embedding,
        "embedding_provider": provider,
        "embedding_device": first_value(
            get_nested(audit, "embedding", "device"),
            metadata.get("embedding_device"),
        ),
        "architecture": architecture,
        "status": first_value(
            report.get("status"),
            metadata.get("status"),
        ),
        "seed": first_value(
            report.get("seed"),
            metadata.get("seed"),
            config.get("seed"),
            get_nested(config, "experiment", "seed"),
        ),
        "protocol": first_value(
            metadata.get("protocol"),
            get_nested(config, "experiment", "protocol"),
            get_nested(config, "dataset", "protocol"),
        ),
        "train_size": first_value(
            metadata.get("train_size"),
            get_nested(audit, "split", "train"),
        ),
        "validation_size": first_value(
            metadata.get("validation_size"),
            get_nested(audit, "split", "validation"),
        ),
        "test_size": first_value(
            metadata.get("test_size"),
            get_nested(audit, "split", "test"),
        ),
        "run_path": str(run),
    }

    for split_name, values, suffix in (
        ("validation", validation, "validation"),
        ("test", test, "test"),
    ):
        for metric in METRIC_ALIASES:
            row[f"{metric}_{suffix}"] = extract_metric(
                values, metric
            )

    history = load_training_history(run)

    training = extract_training(
        run_id,
        dataset,
        embedding,
        architecture,
        history,
    )

    classes = []

    for split, data in (
        ("validation", validation),
        ("test", test),
    ):
        classes.extend(
            extract_class_metrics(
                data,
                run_id,
                dataset,
                embedding,
                architecture,
                split,
            )
        )

    return row, training, classes


def build_internal_comparison(
    experiments: pd.DataFrame,
):
    columns = [
        "dataset",
        "embedding",
        "architecture",
        "protocol",
        "runs",
        "accuracy_test_mean",
        "accuracy_test_std",
        "macro_f1_test_mean",
        "macro_f1_test_std",
        "best_macro_f1_test",
    ]

    if experiments.empty:
        return pd.DataFrame(columns=columns)

    valid = experiments.copy()

    for name in ("accuracy_test", "macro_f1_test"):
        valid[name] = pd.to_numeric(
            valid[name], errors="coerce"
        )

    grouped = valid.groupby(
        [
            "dataset",
            "embedding",
            "architecture",
            "protocol",
        ],
        dropna=False,
    )

    rows = []

    for group_key, subset in grouped:
        dataset, embedding, architecture, protocol = group_key

        rows.append({
            "dataset": dataset,
            "embedding": embedding,
            "architecture": architecture,
            "protocol": protocol,
            "runs": len(subset),
            "accuracy_test_mean": subset[
                "accuracy_test"
            ].mean(),
            "accuracy_test_std": subset[
                "accuracy_test"
            ].std(),
            "macro_f1_test_mean": subset[
                "macro_f1_test"
            ].mean(),
            "macro_f1_test_std": subset[
                "macro_f1_test"
            ].std(),
            "best_macro_f1_test": subset[
                "macro_f1_test"
            ].max(),
        })

    result = pd.DataFrame(rows, columns=columns)

    return result.sort_values(
        ["dataset", "macro_f1_test_mean"],
        ascending=[True, False],
        na_position="last",
    )


def load_baselines(path: Path):
    if not path.is_file():
        return pd.DataFrame(columns=BASELINE_COLUMNS)

    # CSV científico usa o mesmo separador dos arquivos gerados.
    try:
        data = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except (ValueError, OSError, UnicodeError):
        return pd.DataFrame(columns=BASELINE_COLUMNS)

    for column in BASELINE_COLUMNS:
        if column not in data.columns:
            data[column] = None

    return data[BASELINE_COLUMNS]


def build_external_comparison(
    experiments: pd.DataFrame,
    baselines: pd.DataFrame,
):
    rows = []

    if experiments.empty or baselines.empty:
        return pd.DataFrame(columns=EXTERNAL_COLUMNS)

    supported_metrics = {
        "accuracy_test",
        "macro_f1_test",
    }

    for _, baseline in baselines.iterrows():
        verified = str(
            baseline.get("verified", "")
        ).strip().lower()

        if verified not in ("true", "sim", "yes", "1"):
            continue

        dataset = baseline.get("dataset")
        protocol = baseline.get("protocol")
        metric = baseline.get("metric")

        if metric not in supported_metrics:
            continue

        if (
            pd.isna(dataset)
            or pd.isna(protocol)
            or not str(protocol).strip()
        ):
            continue

        published = as_number(baseline.get("value"))

        if published is None:
            continue

        matches = experiments[
            (
                experiments["dataset"].astype(str)
                == str(dataset)
            )
            &
            (
                experiments["protocol"].astype(str)
                == str(protocol)
            )
        ]

        for _, item in matches.iterrows():
            ours = as_number(item.get(metric))

            if ours is None:
                continue

            rows.append({
                "article": baseline.get("article"),
                "dataset": dataset,
                "protocol": protocol,
                "baseline_method": baseline.get("method"),
                "architecture": item.get("architecture"),
                "embedding": item.get("embedding"),
                "metric": metric,
                "our_value": ours,
                "published_value": published,
                "difference": ours - published,
                "source": baseline.get("source"),
                "comparison_status": (
                    "Candidata: verificar equivalência "
                    "de tarefa, partições e protocolo"
                ),
            })

    return pd.DataFrame(rows, columns=EXTERNAL_COLUMNS)


def build_dashboard(experiments: pd.DataFrame):
    columns = [
        "run_id",
        "dataset",
        "embedding",
        "architecture",
        "accuracy_test",
        "macro_f1_test",
        "protocol",
    ]

    if experiments.empty:
        return pd.DataFrame(columns=columns)

    return experiments[columns].sort_values(
        ["dataset", "macro_f1_test"],
        ascending=[True, False],
        na_position="last",
    )


def save_csv(
    data: pd.DataFrame,
    destination: Path,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # UTF-8 com BOM facilita a abertura no Excel Windows.
    # O separador ';' evita conflito com vírgula decimal.
    data.to_csv(
        destination,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Consolida resultados do SNN Research "
            "em arquivos CSV."
        )
    )

    parser.add_argument(
        "--input",
        default="results",
        help="Pasta com os experimentos.",
    )

    parser.add_argument(
        "--output",
        default="results/consolidated",
        help="Pasta de saída dos CSVs.",
    )

    parser.add_argument(
        "--baselines",
        default="references/baselines_scientificos.csv",
        help="Arquivo CSV de baselines verificados.",
    )

    args = parser.parse_args()

    root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    baselines_path = Path(args.baselines).resolve()

    if not root.is_dir():
        raise SystemExit(
            f"Pasta de resultados não encontrada: {root}"
        )

    runs = discover_runs(root, output)

    experiment_records = []
    training_records = []
    class_records = []

    for run in runs:
        try:
            row, training, classes = collect_run(
                run, root
            )

            experiment_records.append(row)
            training_records.append(training)
            class_records.extend(classes)

        except Exception as exc:
            warnings.warn(
                f"Execução ignorada: {run}\nErro: {exc}"
            )

    if not experiment_records:
        raise SystemExit(
            "Nenhuma execução válida encontrada."
        )

    experiments = pd.DataFrame(
        experiment_records,
        columns=EXPERIMENT_COLUMNS,
    )

    classes = pd.DataFrame(
        class_records,
        columns=CLASS_COLUMNS,
    )

    training = pd.DataFrame(
        training_records,
        columns=TRAINING_COLUMNS,
    )

    internal = build_internal_comparison(
        experiments
    )

    baselines = load_baselines(
        baselines_path
    )

    external = build_external_comparison(
        experiments,
        baselines,
    )

    dashboard = build_dashboard(
        experiments
    )

    datasets = {
        "experiments": experiments,
        "classes": classes,
        "training": training,
        "internal": internal,
        "baselines": baselines,
        "external": external,
        "dashboard": dashboard,
    }

    for filename, key in CSV_FILES.items():
        destination = output / filename

        save_csv(
            datasets[key],
            destination,
        )

        print(
            f"[OK] {filename}: "
            f"{len(datasets[key])} registros"
        )

    print("\n" + "=" * 55)
    print("SNN RESEARCH — CONSOLIDAÇÃO FINALIZADA")
    print("=" * 55)

    print(
        f"Execuções identificadas: {len(runs)}"
    )

    print(
        f"Execuções consolidadas: {len(experiments)}"
    )

    print(
        f"Datasets: "
        f"{experiments['dataset'].nunique()}"
    )

    print(
        f"Arquiteturas: "
        f"{experiments['architecture'].nunique()}"
    )

    print(
        f"Embeddings: "
        f"{experiments['embedding'].nunique()}"
    )

    print(f"Saída: {output}")


if __name__ == "__main__":
    main()
