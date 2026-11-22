"""Resolve stand-alone YAML files and modular experiment configurations."""
from copy import deepcopy
from pathlib import Path
import re
import yaml

_COMPONENTS = ("dataset", "embedding", "architecture")
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def _yaml_mapping(path):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")
    return config


def _merge(base, update):
    """Deep-merge mappings without changing the source mappings.

    Lists and scalar values are replaced in full, rather than concatenated.
    """
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path, component_overrides=None):
    """Load both legacy all-in-one YAML and component-based experiments.

    For component-based experiments the resolution order is:
      configs/defaults.yaml -> datasets -> embeddings -> architectures -> experiment overrides.
    The directory containing 'configs/' is inferred from the experiment YAML,
    so calls from a different working directory are also supported.
    """
    path = Path(path).expanduser().resolve()
    experiment = _yaml_mapping(path)
    components = experiment.get("components")
    component_overrides = component_overrides or {}
    if not isinstance(component_overrides, dict):
        raise ValueError("component_overrides must be a mapping")
    if set(component_overrides) - set(_COMPONENTS):
        raise ValueError("Unknown component override; expected dataset, embedding or architecture")
    for category, name in component_overrides.items():
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"Invalid {category} override: {name!r}")
    if components is None:
        if component_overrides:
            raise ValueError("CLI component selection requires a modular 'components:' experiment YAML; legacy complete YAMLs are immutable")
        # Backward compatibility for previously stored stand-alone YAML files.
        # An incomplete or unexpanded modular config must fail before training.
        return _validate_resolved(experiment, path)
    if not isinstance(components, dict):
        raise ValueError("components must be a mapping")
    components = {**components, **component_overrides}
    if set(components) != set(_COMPONENTS):
        raise ValueError(f"components must define exactly: {', '.join(_COMPONENTS)}")
    # Permit organization in configs/experiments/ and nested subdirectories.
    # Resolve the closest configs/ ancestor; never use the working directory.
    config_root = None
    for parent in path.parents:
        if parent.name == "experiments" and parent.parent.name == "configs":
            candidate = parent.parent
            if (candidate / "defaults.yaml").is_file():
                config_root = candidate
                break
    if config_root is None:
        raise ValueError("A modular experiment must be inside configs/experiments/ (including subdirectories) with configs/defaults.yaml")
    resolved = _yaml_mapping(config_root / "defaults.yaml")
    for category in _COMPONENTS:
        name = components[category]
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"Invalid {category} component name: {name!r}")
        source = config_root / (category + 's') / (name + ".yaml")
        # 'architectures', 'embeddings', 'datasets' all follow category+'s',
        # except 'architecture' (spelling changes).
        if category == "architecture":
            source = config_root / "architectures" / (name + ".yaml")
        if not source.is_file():
            raise FileNotFoundError(f"Unknown {category} preset {name!r}: {source}")
        preset = _yaml_mapping(source)
        if category not in preset or not isinstance(preset[category], dict):
            raise ValueError(f"Component preset {source} must define '{category}'")
        resolved = _merge(resolved, preset)
    overrides = {key: value for key, value in experiment.items() if key != 'components'}
    # When components are replaced via CLI, inherited component-specific inline
    # overrides could leak settings from the originally selected dataset/embedding.
    if component_overrides:
        for category in component_overrides:
            overrides.pop(category, None)
        if 'dataset' in component_overrides:
            overrides.pop('split', None)
    resolved = _merge(resolved, overrides)
    resolved['selected_components'] = deepcopy(components)
    protocol = resolved['dataset'].get('split_protocol')
    if protocol and ('dataset' in component_overrides or 'split' not in overrides):
        resolved['split'] = _merge(resolved.get('split', {}), protocol)
    # A stored FAISS index represents a specific dataset, not a generic model.
    scope = resolved['embedding'].get('dataset_scope')
    if scope and scope != components['dataset']:
        raise ValueError(f"FAISS embedding preset {components['embedding']!r} is linked to dataset {scope!r}, not {components['dataset']!r}. Select a model-generated embedding.")
    # Dataset owns the text representation; embedding presets own the model.
    # Do not alter existing FAISS/NPY providers or their alignment safeguards.
    if resolved['embedding'].get('provider', '').lower() in ('model', 'sentence_transformers'):
        input_cfg = resolved['dataset'].get('embedding_input')
        if not input_cfg or not isinstance(input_cfg.get('columns'), list):
            raise ValueError(f"Dataset {components['dataset']!r} needs dataset.embedding_input.columns for model-generated embeddings")
        resolved['embedding']['text_columns'] = deepcopy(input_cfg['columns'])
        resolved['embedding']['separator'] = input_cfg.get('separator', '\n')
        resolved['embedding']['cache_namespace'] = components['dataset']
    if component_overrides:
        # Correct experiment identification for changed combinations.
        selected = resolved['selected_components']
        resolved.setdefault('experiment', {})['name'] = '_'.join(
            (selected['dataset'], selected['embedding'], selected['architecture']))
    return _validate_resolved(resolved, path)


def _validate_resolved(config, source):
    """Fail with a useful message when a modular YAML was not expanded."""
    required = ("experiment", "dataset", "embedding", "model", "training", "split")
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(
            f"Incomplete experiment configuration: {source}. Missing sections: "
            f"{', '.join(missing)}. If this YAML uses 'components:', "
            "update src/utils/config.py together with the configs/ directory "
            "from the same project version."
        )
    for section, field in (("dataset", "path"), ("dataset", "label_column"),
                           ("experiment", "name"), ("experiment", "seed")):
        if config[section].get(field) is None:
            raise ValueError(f"Missing required configuration setting: {section}.{field} ({source})")
    return config


def resolve_path(value, project_root):
    p = Path(value).expanduser()
    return p if p.is_absolute() else Path(project_root) / p