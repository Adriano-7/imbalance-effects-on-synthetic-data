import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._create_attributes(config_dict)

    def _create_attributes(self, config_dict: Dict[str, Any], prefix: str = ""):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        return self._config

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str = None) -> Config:
    if config_path is None:
        project_root = Path(__file__).parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure config.yaml exists in the project root."
        )

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def get_config() -> Config:
    return load_config()


def load_datasets_registry(registry_path: str = None) -> Dict[str, Any]:
    if registry_path is None:
        config_dir = Path(__file__).parent
        registry_path = config_dir / "datasets.yaml"
    else:
        registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Dataset registry not found: {registry_path}\n"
            f"Please ensure config/datasets.yaml exists."
        )

    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)

    return registry.get('datasets', {})


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    datasets = load_datasets_registry()

    if dataset_name not in datasets:
        available = ', '.join(datasets.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry.\n"
            f"Available datasets: {available}"
        )

    return datasets[dataset_name]


def list_available_datasets() -> list:
    datasets = load_datasets_registry()
    return list(datasets.keys())


def validate_dataset_exists(dataset_name: str) -> bool:
    datasets = load_datasets_registry()
    return dataset_name in datasets


if __name__ == "__main__":
    config = get_config()

    print(f"Imbalance Ratios: {config.experiment.imbalance_ratios}")
    print(f"Number of Repetitions: {config.experiment.n_repetitions}")
    print(f"Generators: {config.models.generators}")

    print(f"Random State: {config.get('experiment.random_state', 42)}")
    print(f"Parallel Jobs: {config.execution.parallel.n_jobs}")

    print("\n  Dataset Registry  ")
    print(f"Available datasets: {list_available_datasets()}")

    for dataset_name in list_available_datasets():
        dataset_config = get_dataset_config(dataset_name)
        print(f"\n{dataset_name}:")
        print(f"  Target column: {dataset_config['target_column']}")
        print(f"  Processed path: {dataset_config['processed_path']}")
