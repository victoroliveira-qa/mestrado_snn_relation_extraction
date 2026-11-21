"""Module composition must not change the parameters of prior experiments."""
from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs'


def test_preset_composition_preserves_prior_experiment_sections():
    for old in sorted((CONFIGS / 'legacy').glob('*.yaml')):
        original = load_config(old)
        locations = list((CONFIGS / 'experiments').rglob(old.name))
        assert len(locations) == 1, f'Missing or duplicate modular configuration for {old.name}'
        resolved = load_config(locations[0])
        # Prior model/training/split/embedding values must remain unchanged.
        for section, value in original.items():
            if section == 'dataset':
                assert all(resolved[section][k] == v for k, v in value.items())
            else:
                assert resolved[section] == value, f'{old.name}: {section}'


def test_switching_component_only_affects_selected_section(tmp_path):
    cfg = yaml.safe_load((CONFIGS / 'experiments' / 'business_mlp.yaml').read_text())
    original = load_config(CONFIGS / 'experiments' / 'business_mlp.yaml')
    path = CONFIGS / 'experiments' / 'test_temporary_switch.yaml'
    try:
        cfg['components']['embedding'] = 'bge_m3'
        path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        changed = load_config(path)
        assert changed['embedding']['model'] == 'bge-m3'
        for section in ('dataset', 'model', 'training', 'split', 'architecture'):
            assert original[section] == changed[section]
    finally:
        path.unlink(missing_ok=True)


def test_nested_experiment_override_preserves_unmodified_keys():
    path = CONFIGS / 'experiments' / 'test_temporary_override.yaml'
    cfg = {'components': {'dataset': 'business', 'embedding': 'qwen8b_faiss', 'architecture': 'snn_mlp'},
           'experiment': {'name':'temporary', 'seed': 12},
           'training': {'epochs': 25}, 'model': {'dropout': 0.42}}
    try:
        path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        resolved = load_config(path)
        assert resolved['training']['epochs'] == 25
        assert resolved['training']['batch_size'] == 64
        assert resolved['model']['dropout'] == 0.42
        assert resolved['model']['projection_dims'] == [1024, 256]
        assert resolved['architecture']['name'] == 'snn_mlp'
    finally:
        path.unlink(missing_ok=True)


def test_rejects_unknown_or_unsafe_presets():
    path = CONFIGS / 'experiments' / 'test_temporary_bad.yaml'
    base = {'components': {'dataset':'business','embedding':'qwen8b_faiss','architecture':'not_real'}}
    try:
        path.write_text(yaml.safe_dump(base), encoding='utf-8')
        with pytest.raises(FileNotFoundError):
            load_config(path)
        base['components']['architecture'] = '../legacy/v3_qwen_business'
        path.write_text(yaml.safe_dump(base), encoding='utf-8')
        with pytest.raises(ValueError):
            load_config(path)
    finally:
        path.unlink(missing_ok=True)


def test_legacy_standalone_yaml_supported():
    old = CONFIGS / 'legacy' / 'v3_qwen_business.yaml'
    parsed = load_config(old)
    assert parsed['embedding']['provider'] == 'faiss'
    assert parsed['architecture']['name'] == 'snn_prototype'
