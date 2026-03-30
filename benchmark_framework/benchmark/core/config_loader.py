from pathlib import Path

import yaml


class ConfigLoader:
    REQUIRED_TOP_LEVEL_KEYS = [
        "run",
        "task",
        "model",
        "backend",
        "dataset",
        "pipeline",
        "benchmark",
        "profiling",
        "monitoring",
        "reporting",
    ]

    @classmethod
    def load(cls, config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Config root must be a dictionary")

        cls._validate(config)
        cls._normalize_run_paths(config)
        return config

    @classmethod
    def _validate(cls, config: dict) -> None:
        for key in cls.REQUIRED_TOP_LEVEL_KEYS:
            if key not in config:
                raise ValueError(f"Missing required config section: '{key}'")

        if "name" not in config["run"]:
            raise ValueError("Missing 'run.name'")

        # Yeni şema: output_root
        # Eski şema: output_dir
        if "output_root" not in config["run"] and "output_dir" not in config["run"]:
            raise ValueError("Missing 'run.output_root' or legacy 'run.output_dir'")

        if "type" not in config["task"]:
            raise ValueError("Missing 'task.type'")

        if "type" not in config["backend"]:
            raise ValueError("Missing 'backend.type'")

        if "type" not in config["dataset"]:
            raise ValueError("Missing 'dataset.type'")

    @classmethod
    def _normalize_run_paths(cls, config: dict) -> None:
        run_cfg = config["run"]

        # Eski config kullanılıyorsa otomatik dönüştür
        if "output_root" not in run_cfg and "output_dir" in run_cfg:
            legacy_output_dir = run_cfg["output_dir"]
            run_cfg["output_root"] = legacy_output_dir

        # global_summary_path yoksa varsayılan ver
        if "global_summary_path" not in run_cfg:
            run_cfg["global_summary_path"] = "./outputs/comparisons/all_runs_summary.csv"