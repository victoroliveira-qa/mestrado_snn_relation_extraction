"""Immutable, timestamped experiment folders and plain-text reports."""
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.utils.output import output_dirs


def create_run(base, experiment_name, now=None):
    when = now or datetime.now().astimezone()
    stamp = when.strftime('%Y-%m-%d_%H-%M-%S_%f')
    suffix = uuid4().hex[:8]
    path = Path(base) / experiment_name / f'{stamp}_{suffix}'
    path.mkdir(parents=True, exist_ok=False)
    return path, output_dirs(path)


def write_txt_report(dirs, config, metrics_by_split, run_path, status='CONCLUIDO', extra=None, audit=None):
    """Highlight a predeclared selection metric within each task, never maximum of unlike metrics."""
    architecture = config.get('architecture', {}).get('name', 'snn_prototype')
    task = 'pair' if architecture in ('snn', 'snn_mlp') else 'multiclass'
    metric = config.get('report', {}).get('primary_metric') or ('pair_f1' if task == 'pair' else 'macro_f1')
    stamp = run_path.name.split('_')[0] + '_' + run_path.name.split('_')[1]
    filename = f'relatorio_{stamp}_{run_path.name.split("_")[-1]}.txt'
    parts = ['SNN RESEARCH — RELATORIO EXPERIMENTAL', '='*55,
             f'Execucao: {run_path.name}', f'Status: {status}',
             f'Arquitetura: {architecture}', f'Dataset: {config.get("dataset", {}).get("path", "NAO RESOLVIDO")}',
             f'Embedding: {json.dumps(config.get("embedding", "NAO RESOLVIDO"), ensure_ascii=False)}',
             f'Seed: {config.get("experiment", {}).get("seed", "NAO INFORMADA")}',
             f'Metrica principal (predefinida): {metric}', '',
             'METRICAS POR CONJUNTO', '-'*55]
    for split, metrics in metrics_by_split.items():
        parts.append(f'[{split.upper()}]')
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                parts.append(f'{"*** " if k == metric else "    "}{k}: {v:.6f}{"  <-- METRICA PRINCIPAL" if k == metric else ""}')
        if metric in metrics:
            parts.append(f'DESTAQUE {split.upper()}: {metric} = {metrics[metric]:.6f}')
        parts.append('')
    if audit:
        from src.evaluation.audit import report_audit_lines, report_class_lines
        parts.extend(report_audit_lines(audit))
        parts.extend(report_class_lines(metrics_by_split))
    if extra:
        parts.extend(['OBSERVACOES', '-'*55] + [str(x) for x in extra])
    parts.extend(['', 'ARQUIVOS', '-'*55] + [f'{key}/' for key in dirs])
    target = dirs['reports'] / filename
    # Exclusive create explicitly avoids accidental overwrite.
    with target.open('x', encoding='utf-8') as file:
        file.write('\n'.join(parts) + '\n')
    return target
