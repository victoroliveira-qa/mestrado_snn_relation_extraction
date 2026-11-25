from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "run_id",
    "data_registro_utc",
    "experimento",
    "dataset",
    "arquitetura",
    "embedding_modelo",
    "embedding_provedor",
    "embedding_dispositivo",
    "status",
    "seed",
    "protocolo_json",
    "treino_registros",
    "validacao_registros",
    "teste_registros",
    "accuracy_validacao",
    "f1_macro_validacao",
    "accuracy_teste",
    "f1_macro_teste",
    "epocas_executadas",
    "loss_final",
    "melhor_val_loss",
    "melhor_epoca",
    "duracao_segundos",
    "metricas_validacao_json",
    "metricas_teste_json",
    "metricas_adicionais_json",
    "metricas_por_classe_json",
    "historico_treinamento_json",
    "configuracao_completa_json",
    "auditoria_json",
    "metadados_execucao_json",
    "informacoes_runtime_json",
    "relatorios_individuais_json",
    "manifesto_artefatos_json",
    "pasta_execucao",
]

ARCHITECTURES = [
    "snn_prototype",
    "mlp_prototype",
    "snn_mlp",
    "prototype",
    "snn",
    "mlp",
]


def safe(value):
    """Converte objetos para valores serializáveis em JSON."""
    if isinstance(value, Mapping):
        return {
            str(key): safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    if hasattr(value, "item"):
        try:
            return safe(value.item())
        except (TypeError, ValueError):
            pass

    return str(value)


def dumps(value):
    return json.dumps(
        safe(value),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def pretty_json(value):
    return json.dumps(
        safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )


def get(data, *keys):
    value = data

    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)

    return value


def first(*values):
    for value in values:
        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, (dict, list)) and not value:
            continue

        return value

    return None


def number(value):
    try:
        if value is None or isinstance(value, bool):
            return None

        result = float(str(value).replace(",", "."))

        return result if math.isfinite(result) else None

    except (TypeError, ValueError):
        return None


def read_json(path):
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError, UnicodeError):
        return None


def read_text(path):
    try:
        return path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )
    except OSError:
        return None


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def collect_artifacts(run_dir):
    """
    Inventaria os arquivos da execução.

    Arquivos pequenos recebem SHA-256.
    Arquivos volumosos são referenciados sem cópia.
    """
    manifest = []

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(run_dir).as_posix()

        try:
            size = path.stat().st_size
        except OSError:
            continue

        item = {
            "arquivo": relative,
            "tamanho_bytes": size,
            "sha256": None,
        }

        if (
            size <= 10 * 1024 * 1024
            and path.suffix.lower() in {
                ".json", ".csv", ".yaml",
                ".yml", ".txt",
            }
        ):
            try:
                item["sha256"] = sha256(path)
            except OSError:
                pass

        manifest.append(item)

    return manifest


def collect_documents(run_dir):
    """
    Captura dados textuais e estruturados disponíveis.

    Não duplica modelos, embeddings ou predições extensas.
    Os arquivos omitidos permanecem no manifesto.
    """
    documents = {}
    reports = {}

    allowed_folders = {
        "configs",
        "metrics",
        "logs",
        "reports",
        "splits",
    }

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(run_dir)
        relative_key = relative.as_posix()

        if (
            len(relative.parts) > 1
            and relative.parts[0] not in allowed_folders
        ):
            continue

        if path.stat().st_size > 5 * 1024 * 1024:
            continue

        suffix = path.suffix.lower()

        if suffix == ".json":
            data = read_json(path)

            if data is not None:
                documents[relative_key] = data

        elif suffix in {".yaml", ".yml", ".txt"}:
            content = read_text(path)

            if content is None:
                continue

            if relative.parts[0] == "reports":
                reports[relative_key] = content
            else:
                documents[relative_key] = content

    return documents, reports


def read_history(run_dir):
    """Lê os históricos CSV sem exigir Pandas."""
    histories = {}

    candidates = [
        run_dir / "metrics" / "training_history.csv",
        run_dir / "logs" / "training_history.csv",
        run_dir / "training_history.csv",
        run_dir / "metrics" / "history.csv",
    ]

    for path in candidates:
        if not path.is_file():
            continue

        try:
            with path.open(
                encoding="utf-8-sig",
                newline="",
            ) as file:
                sample = file.read(4096)
                file.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(
                        sample,
                        delimiters=",;\t",
                    )
                except csv.Error:
                    dialect = csv.excel

                rows = list(
                    csv.DictReader(file, dialect=dialect)
                )

            histories[
                path.relative_to(run_dir).as_posix()
            ] = rows

        except (OSError, UnicodeError, csv.Error):
            continue

    return histories


def report_field(text, field):
    match = re.search(
        rf"(?im)^\s*{re.escape(field)}:\s*(.+?)\s*$",
        text,
    )
    return match.group(1).strip() if match else None


def report_metrics(text, split):
    """Recupera métricas de relatórios TXT antigos."""
    match = re.search(
        rf"(?ims)^\[{re.escape(split)}\]\s*"
        r"(.*?)(?=^\[|^ARQUIVOS|\Z)",
        text,
    )

    if not match:
        return {}

    values = {}

    for name, raw in re.findall(
        r"(?im)\b(accuracy|macro_f1|f1_macro)"
        r"\s*:\s*([0-9]+(?:[.,][0-9]+)?)",
        match.group(1),
    ):
        values[name.lower()] = number(raw)

    return values


def metric(data, *aliases):
    if not isinstance(data, dict):
        return None

    sources = [
        data,
        data.get("metrics"),
        data.get("test"),
        data.get("validation"),
    ]

    for source in sources:
        if not isinstance(source, dict):
            continue

        for alias in aliases:
            value = number(source.get(alias))

            if value is not None:
                return value

    return None


def classification_report(data):
    if not isinstance(data, dict):
        return {}

    sources = [
        data,
        data.get("metrics"),
        data.get("test"),
        data.get("validation"),
    ]

    for source in sources:
        if not isinstance(source, dict):
            continue

        for name in (
            "classification_report",
            "per_class",
            "class_report",
        ):
            value = source.get(name)

            if isinstance(value, dict):
                return value

    return {}


def support_total(data):
    """Recupera tamanho do conjunto quando há suporte por classe."""
    report = classification_report(data)

    ignored = {
        "accuracy",
        "macro avg",
        "weighted avg",
        "micro avg",
        "samples avg",
    }

    supports = [
        number(values.get("support"))
        for label, values in report.items()
        if label not in ignored
        and isinstance(values, dict)
    ]

    if not supports or any(v is None for v in supports):
        return None

    return int(sum(supports))


def extract_history_stats(histories):
    result = {
        "epocas_executadas": None,
        "loss_final": None,
        "melhor_val_loss": None,
        "melhor_epoca": None,
    }

    if not histories:
        return result

    rows = next(iter(histories.values()))

    if not rows:
        return result

    result["epocas_executadas"] = len(rows)

    for field in ("loss", "train_loss", "training_loss"):
        value = number(rows[-1].get(field))

        if value is not None:
            result["loss_final"] = value
            break

    valid = []

    for index, row in enumerate(rows, start=1):
        loss = first(
            number(row.get("val_loss")),
            number(row.get("validation_loss")),
        )

        if loss is not None:
            valid.append((loss, index, row))

    if valid:
        best_loss, best_index, best_row = min(
            valid, key=lambda item: item[0]
        )

        result["melhor_val_loss"] = best_loss
        result["melhor_epoca"] = first(
            best_row.get("epoch"),
            best_index,
        )

    return result


def infer_architecture(config, report, experiment):
    architecture = first(
        get(config, "architecture", "name"),
        get(config, "architecture", "type"),
        report_field(report, "Arquitetura"),
    )

    if architecture is not None:
        return str(architecture)

    experiment = experiment.lower()

    for name in ARCHITECTURES:
        if experiment.endswith(name):
            return name

    return None


def load_config_snapshot(documents):
    """
    Usa a configuração resolvida salva pela execução,
    quando o chamador não forneceu cfg diretamente.
    """
    preferred = (
        "configs/resolved_config.yaml",
        "configs/config.yaml",
        "configs/experiment.yaml",
        "configs/resolved.yaml",
    )

    try:
        import yaml
    except ImportError:
        return {}

    for path in preferred:
        content = documents.get(path)

        if not isinstance(content, str):
            continue

        try:
            value = yaml.safe_load(content)

            if isinstance(value, dict):
                return value
        except yaml.YAMLError:
            pass

    return {}


def build_record(
    run_dir,
    resolved_config=None,
    runtime_extra=None,
    status="CONCLUIDO",
):
    run_dir = Path(run_dir).resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Pasta da execução não encontrada: {run_dir}"
        )

    documents, reports = collect_documents(run_dir)
    histories = read_history(run_dir)

    config = first(
        resolved_config,
        load_config_snapshot(documents),
        {},
    )

    if not isinstance(config, dict):
        config = {"configuracao": config}

    runtime_extra = safe(runtime_extra or {})

    metadata = first(
        documents.get("metadata.json"),
        documents.get("logs/metadata.json"),
        documents.get("metrics/metadata.json"),
        {},
    )

    audit = first(
        documents.get("logs/experimental_audit.json"),
        {},
    )

    test = first(
        documents.get("metrics/test_metrics.json"),
        documents.get("metrics/metrics_test.json"),
        {},
    )

    validation = first(
        documents.get("metrics/validation_metrics.json"),
        documents.get("metrics/val_metrics.json"),
        {},
    )

    report_text = (
        next(iter(reports.values()))
        if reports else ""
    )

    if not test:
        test = report_metrics(report_text, "TEST")

    if not validation:
        validation = report_metrics(
            report_text, "VALIDATION"
        )

    experiment = run_dir.parent.name
    run_id = f"{experiment}/{run_dir.name}"

    dataset_config = config.get("dataset", {})
    embedding_config = config.get("embedding", {})

    if not isinstance(dataset_config, dict):
        dataset_config = {"name": dataset_config}

    if not isinstance(embedding_config, dict):
        embedding_config = {
            "model": embedding_config
        }

    dataset = first(
        get(metadata, "dataset", "name"),
        dataset_config.get("name"),
        dataset_config.get("path"),
        report_field(report_text, "Dataset"),
    )

    embedding_raw = first(
        embedding_config.get("model"),
        embedding_config.get("name"),
        get(metadata, "embedding", "model"),
    )

    embedding_provider = first(
        embedding_config.get("provider"),
        get(metadata, "embedding", "provider"),
    )

    # Não presume que provider='faiss'
    # corresponda necessariamente ao Qwen8B.
    if embedding_raw is None:
        embedding_raw = report_field(
            report_text, "Embedding"
        )

        if (
            isinstance(embedding_raw, str)
            and embedding_raw.startswith("{")
        ):
            try:
                embedding_raw = json.loads(embedding_raw)
            except ValueError:
                pass

    if isinstance(embedding_raw, dict):
        embedding_provider = first(
            embedding_provider,
            embedding_raw.get("provider"),
        )
        embedding_raw = first(
            embedding_raw.get("model"),
            embedding_raw.get("name"),
        )

    history_stats = extract_history_stats(histories)

    protocol = first(
        dataset_config.get("split_protocol"),
        dataset_config.get("protocol"),
        get(metadata, "split_protocol"),
        get(audit, "split_protocol"),
    )

    split_info = first(
        get(audit, "split"),
        get(audit, "split_counts"),
        get(metadata, "split_counts"),
        {},
    )

    if not isinstance(split_info, dict):
        split_info = {}

    class_metrics = {
        "validation": classification_report(validation),
        "test": classification_report(test),
    }

    other_metrics = {
        name: content
        for name, content in documents.items()
        if name.startswith("metrics/")
        and name.endswith(".json")
        and name not in {
            "metrics/test_metrics.json",
            "metrics/validation_metrics.json",
            "metrics/val_metrics.json",
        }
    }

    return {
        "run_id": run_id,
        "data_registro_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "experimento": experiment,
        "dataset": dataset,
        "arquitetura": infer_architecture(
            config, report_text, experiment
        ),
        "embedding_modelo": embedding_raw,
        "embedding_provedor": embedding_provider,
        "embedding_dispositivo": first(
            get(audit, "embedding", "device"),
            get(audit, "embedding", "generation_device"),
            get(metadata, "embedding", "device"),
            runtime_extra.get("embedding_device"),
        ),
        "status": first(
            report_field(report_text, "Status"),
            status,
        ),
        "seed": first(
            report_field(report_text, "Seed"),
            get(config, "experiment", "seed"),
            config.get("seed"),
            metadata.get("seed"),
        ),
        "protocolo_json": dumps(protocol),
        "treino_registros": first(
            metadata.get("train_size"),
            split_info.get("train"),
            runtime_extra.get("train_size"),
        ),
        "validacao_registros": first(
            metadata.get("validation_size"),
            split_info.get("validation"),
            support_total(validation),
        ),
        "teste_registros": first(
            metadata.get("test_size"),
            split_info.get("test"),
            support_total(test),
        ),
        "accuracy_validacao": metric(
            validation, "accuracy", "acc"
        ),
        "f1_macro_validacao": metric(
            validation, "macro_f1", "f1_macro"
        ),
        "accuracy_teste": metric(
            test, "accuracy", "acc"
        ),
        "f1_macro_teste": metric(
            test, "macro_f1", "f1_macro"
        ),
        **history_stats,
        "duracao_segundos": first(
            metadata.get("duration_seconds"),
            metadata.get("elapsed_seconds"),
            runtime_extra.get("duration_seconds"),
        ),
        "metricas_validacao_json": dumps(validation),
        "metricas_teste_json": dumps(test),
        "metricas_adicionais_json": dumps(other_metrics),
        "metricas_por_classe_json": dumps(class_metrics),
        "historico_treinamento_json": dumps(histories),
        "configuracao_completa_json": dumps({
            "resolved_config": config,
            "saved_documents": {
                key: value
                for key, value in documents.items()
                if key.startswith("configs/")
            },
        }),
        "auditoria_json": dumps(audit),
        "metadados_execucao_json": dumps(metadata),
        "informacoes_runtime_json": dumps(runtime_extra),
        "relatorios_individuais_json": dumps(reports),
        "manifesto_artefatos_json": dumps(
            collect_artifacts(run_dir)
        ),
        "pasta_execucao": str(run_dir),
    }


@contextmanager
def registry_lock(output_dir):
    """
    Bloqueio exclusivo para evitar concorrência
    entre processos de treinamento.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".registry.lock"

    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)

        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()

        handle.seek(0)

        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        try:
            yield
        finally:
            handle.seek(0)

            if os.name == "nt":
                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
            else:
                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_UN,
                )


def atomic_write(path, content):
    """
    Grava um arquivo temporário e substitui
    o destino de forma atômica.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            delete=False,
            dir=path.parent,
            prefix=".tmp_registry_",
        ) as file:
            file.write(content)
            temporary = Path(file.name)

        os.replace(temporary, path)

    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def csv_content(records):
    import io

    buffer = io.StringIO(newline="")

    writer = csv.DictWriter(
        buffer,
        fieldnames=FIELDS,
        delimiter=";",
        extrasaction="ignore",
    )

    writer.writeheader()

    for record in records:
        writer.writerow(record)

    return buffer.getvalue()


def txt_content(records):
    lines = [
        "SNN RESEARCH — REGISTRO EXPERIMENTAL GLOBAL",
        "=" * 74,
        f"Quantidade de execuções: {len(records)}",
        "",
    ]

    detailed_fields = {
        "metricas_validacao_json",
        "metricas_teste_json",
        "metricas_adicionais_json",
        "metricas_por_classe_json",
        "historico_treinamento_json",
        "configuracao_completa_json",
        "auditoria_json",
        "metadados_execucao_json",
        "informacoes_runtime_json",
        "relatorios_individuais_json",
        "manifesto_artefatos_json",
        "protocolo_json",
    }

    for index, record in enumerate(records, start=1):
        lines.extend([
            "",
            "=" * 74,
            f"EXECUÇÃO {index}: {record['run_id']}",
            "=" * 74,
        ])

        for field in FIELDS:
            if field in detailed_fields:
                continue

            lines.append(
                f"{field}: {record.get(field, '')}"
            )

        for field in FIELDS:
            if field not in detailed_fields:
                continue

            value = record.get(field, "")

            try:
                parsed = json.loads(value) if value else None
                formatted = pretty_json(parsed)
            except (ValueError, TypeError):
                formatted = str(value)

            lines.extend([
                "",
                "-" * 74,
                field.upper(),
                "-" * 74,
                formatted,
            ])

    return "\n".join(lines) + "\n"


def register_execution(
    run_dir,
    resolved_config=None,
    runtime_extra=None,
    status="CONCLUIDO",
    output_dir="results/consolidated",
):
    """
    Registra uma execução nos arquivos gerais.

    Deve ser chamado APÓS a gravação dos relatórios
    e métricas individuais.

    Execuções já existentes são atualizadas pelo run_id,
    evitando duplicações.
    """
    record = build_record(
        run_dir=run_dir,
        resolved_config=resolved_config,
        runtime_extra=runtime_extra,
        status=status,
    )

    output_dir = Path(output_dir).resolve()

    csv_path = output_dir / "resultados_gerais.csv"
    txt_path = output_dir / "resultados_gerais.txt"

    with registry_lock(output_dir):
        previous = {}

        if csv_path.is_file():
            with csv_path.open(
                encoding="utf-8-sig",
                newline="",
            ) as file:
                reader = csv.DictReader(
                    file,
                    delimiter=";",
                )

                for row in reader:
                    run_id = row.get("run_id")

                    if run_id:
                        previous[run_id] = row

        # UPSERT: substitui o registro da mesma execução.
        # Não cria linhas duplicadas.
        previous[record["run_id"]] = record

        records = list(previous.values())

        atomic_write(
            csv_path,
            csv_content(records),
        )

        # Reconstitui o TXT com os registros preservados.
        # Isso evita blocos duplicados após reprocessamentos.
        atomic_write(
            txt_path,
            txt_content(records),
        )

    print(
        f"[REGISTRO GLOBAL] Execução registrada: "
        f"{record['run_id']}"
    )
    print(f"[REGISTRO GLOBAL] CSV: {csv_path}")
    print(f"[REGISTRO GLOBAL] TXT: {txt_path}")

    return record


def backfill_results(
    results_dir="results",
    output_dir="results/consolidated",
):
    """
    Importa resultados antigos uma única vez.

    Informações indisponíveis permanecem ausentes,
    sem reconstrução por suposição.
    """
    root = Path(results_dir).resolve()
    output = Path(output_dir).resolve()

    runs = set()

    for reports_dir in root.rglob("reports"):
        if reports_dir.is_dir():
            runs.add(reports_dir.parent)

    for metrics_file in root.rglob("test_metrics.json"):
        if metrics_file.parent.name == "metrics":
            runs.add(metrics_file.parent.parent)

    imported = 0

    for run in sorted(runs):
        if run == output or output in run.parents:
            continue

        try:
            register_execution(
                run_dir=run,
                output_dir=output,
            )
            imported += 1
        except Exception as exc:
            print(f"[ERRO] {run}: {exc}")

    print(f"\nExecuções processadas: {imported}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backfill",
        default="results",
        help="Pasta de resultados antigos.",
    )

    parser.add_argument(
        "--output",
        default="results/consolidated",
        help="Pasta dos registros gerais.",
    )

    args = parser.parse_args()

    backfill_results(
        results_dir=args.backfill,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
