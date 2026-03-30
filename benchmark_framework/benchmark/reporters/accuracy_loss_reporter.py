from __future__ import annotations

import csv
import json
import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class MetricBundle:
    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None

    def is_valid(self) -> bool:
        return any(v is not None for v in [
            self.precision, self.recall, self.map50, self.map50_95
        ])


@dataclass
class RunRecord:
    run_name: str
    run_dir: str
    report_path: str
    config_path: Optional[str]

    model_name: Optional[str]
    backend_type: Optional[str]
    dataset_name: Optional[str]
    precision_mode: Optional[str]
    model_path: Optional[str]
    input_size: Optional[str]
    pairing_key: str

    metrics: MetricBundle


@dataclass
class AccuracyLossRecord:
    pairing_key: str
    model_name: Optional[str]
    backend_type: Optional[str]
    dataset_name: Optional[str]
    input_size: Optional[str]

    fp16_run_name: Optional[str]
    int8_run_name: Optional[str]

    fp16_report_path: Optional[str]
    int8_report_path: Optional[str]

    fp16_precision_metric: Optional[float]
    int8_precision_metric: Optional[float]
    precision_loss: Optional[float]

    fp16_recall: Optional[float]
    int8_recall: Optional[float]
    recall_loss: Optional[float]

    fp16_map50: Optional[float]
    int8_map50: Optional[float]
    map50_loss: Optional[float]

    fp16_map50_95: Optional[float]
    int8_map50_95: Optional[float]
    map50_95_loss: Optional[float]


def _safe_get(data: Dict[str, Any], *keys: str, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_precision(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    mapping = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "int8": "int8",
        "i8": "int8",
        "fp32": "fp32",
        "float32": "fp32",
        "float": "fp32",
        "full": "fp32",
    }
    return mapping.get(v, v)


def _normalize_input_size(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "x".join(str(x) for x in value)
    return str(value)


def _hash_dict(d: Dict[str, Any]) -> str:
    raw = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _delta(fp16_value: Optional[float], int8_value: Optional[float]) -> Optional[float]:
    if fp16_value is None or int8_value is None:
        return None
    return fp16_value - int8_value


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _infer_precision_from_model_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None

    name = Path(model_path).name.lower()

    if re.search(r"(^|[_\-.])fp16([_\-.]|$)", name):
        return "fp16"
    if re.search(r"(^|[_\-.])int8([_\-.]|$)", name):
        return "int8"
    if re.search(r"(^|[_\-.])fp32([_\-.]|$)", name):
        return "fp32"

    return None


def _infer_backend_from_model_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None

    suffix = Path(model_path).suffix.lower()
    mapping = {
        ".pt": "pytorch",
        ".onnx": "onnxruntime",
        ".engine": "tensorrt",
        ".rknn": "rknn",
    }
    return mapping.get(suffix)


def _infer_model_name_from_model_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None

    stem = Path(model_path).stem

    # precision suffix'lerini temizle
    stem = re.sub(r"([_\-.])(fp16|fp32|int8)$", "", stem, flags=re.IGNORECASE)

    return stem


def _extract_metrics(report: Dict[str, Any]) -> MetricBundle:
    evaluation_summary = report.get("evaluation_summary", {})

    return MetricBundle(
        precision=_to_float(evaluation_summary.get("precision")),
        recall=_to_float(evaluation_summary.get("recall")),
        map50=_to_float(evaluation_summary.get("map_50")),
        map50_95=_to_float(evaluation_summary.get("map_50_95")),
    )


def _extract_metadata(report: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    run_metadata = report.get("run_metadata", {})
    config_model = config.get("model", {}) if isinstance(config, dict) else {}
    config_backend = config.get("backend", {}) if isinstance(config, dict) else {}
    config_dataset = config.get("dataset", {}) if isinstance(config, dict) else {}

    run_name = run_metadata.get("run_name") or _safe_get(config, "run", "name")

    model_path = (
        run_metadata.get("model_path")
        or config_model.get("path")
    )

    precision_mode = (
        run_metadata.get("precision")
        or config_model.get("precision")
        or _infer_precision_from_model_path(model_path)
    )
    precision_mode = _normalize_precision(precision_mode)

    backend_type = (
        run_metadata.get("backend")
        or config_backend.get("type")
        or _infer_backend_from_model_path(model_path)
    )

    model_name = (
        run_metadata.get("model_name")
        or config_model.get("name")
        or _infer_model_name_from_model_path(model_path)
    )

    dataset_name = (
        run_metadata.get("dataset_name")
        or config_dataset.get("name")
        or config_dataset.get("type")
        or "unknown_dataset"
    )

    input_size = (
        run_metadata.get("input_size")
        or config_model.get("input_size")
    )

    return {
        "run_name": run_name,
        "model_path": model_path,
        "precision_mode": precision_mode,
        "backend_type": backend_type,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "input_size": _normalize_input_size(input_size),
    }


def _build_pairing_key(meta: Dict[str, Any]) -> str:
    key_payload = {
        "model_name": meta.get("model_name"),
        "backend_type": meta.get("backend_type"),
        "dataset_name": meta.get("dataset_name"),
        "input_size": meta.get("input_size"),
    }

    readable = "|".join([
        str(meta.get("model_name") or "na"),
        str(meta.get("backend_type") or "na"),
        str(meta.get("dataset_name") or "na"),
        str(meta.get("input_size") or "na"),
    ])
    digest = _hash_dict(key_payload)
    return f"{readable}|{digest}"


def _pick_best_run(records: List[RunRecord], target_precision: str) -> Optional[RunRecord]:
    filtered = [r for r in records if r.precision_mode == target_precision]
    if not filtered:
        return None

    filtered.sort(
        key=lambda r: (
            r.metrics.map50_95 if r.metrics.map50_95 is not None else -1.0,
            r.metrics.map50 if r.metrics.map50 is not None else -1.0,
            r.metrics.precision if r.metrics.precision is not None else -1.0,
            r.metrics.recall if r.metrics.recall is not None else -1.0,
        ),
        reverse=True,
    )
    return filtered[0]


class AccuracyLossReporter:
    def __init__(self, run_dirs: List[str]):
        self.run_dirs = [Path(p) for p in run_dirs]

    def _load_run_record(self, run_dir: Path) -> Optional[RunRecord]:
        report_path = run_dir / "report.json"
        config_path = run_dir / "config_snapshot.yaml"

        if not report_path.exists():
            return None

        try:
            report = _load_json(report_path)
        except Exception as e:
            print(f"[WARN] report.json okunamadı: {report_path} -> {e}")
            return None

        config = {}
        if config_path.exists():
            try:
                config = _load_yaml(config_path)
            except Exception as e:
                print(f"[WARN] config_snapshot.yaml okunamadı: {config_path} -> {e}")

        meta = _extract_metadata(report, config)
        metrics = _extract_metrics(report)

        pairing_key = _build_pairing_key(meta)

        return RunRecord(
            run_name=meta.get("run_name") or run_dir.name,
            run_dir=str(run_dir),
            report_path=str(report_path),
            config_path=str(config_path) if config_path.exists() else None,

            model_name=meta.get("model_name"),
            backend_type=meta.get("backend_type"),
            dataset_name=meta.get("dataset_name"),
            precision_mode=meta.get("precision_mode"),
            model_path=meta.get("model_path"),
            input_size=meta.get("input_size"),
            pairing_key=pairing_key,
            metrics=metrics,
        )

    def collect_runs(self) -> List[RunRecord]:
        records: List[RunRecord] = []

        for run_dir in self.run_dirs:
            if not run_dir.exists() or not run_dir.is_dir():
                print(f"[WARN] invalid run dir: {run_dir}")
                continue

            rec = self._load_run_record(run_dir)
            if rec is None:
                continue

            if rec.precision_mode not in {"fp16", "int8", "fp32"}:
                print(f"[INFO] precision tespit edilemedi, skip: {rec.run_name}")
                continue

            if not rec.metrics.is_valid():
                print(f"[INFO] metric bulunamadı, skip: {rec.run_name}")
                continue

            records.append(rec)

        return records

    def build_accuracy_loss_records(self, runs: List[RunRecord]) -> List[AccuracyLossRecord]:
        grouped: Dict[str, List[RunRecord]] = {}
        for run in runs:
            grouped.setdefault(run.pairing_key, []).append(run)

        results: List[AccuracyLossRecord] = []

        for pairing_key, group in grouped.items():
            fp16_run = _pick_best_run(group, "fp16")
            int8_run = _pick_best_run(group, "int8")

            if fp16_run is None or int8_run is None:
                continue

            results.append(
                AccuracyLossRecord(
                    pairing_key=pairing_key,
                    model_name=fp16_run.model_name or int8_run.model_name,
                    backend_type=fp16_run.backend_type or int8_run.backend_type,
                    dataset_name=fp16_run.dataset_name or int8_run.dataset_name,
                    input_size=fp16_run.input_size or int8_run.input_size,

                    fp16_run_name=fp16_run.run_name,
                    int8_run_name=int8_run.run_name,

                    fp16_report_path=fp16_run.report_path,
                    int8_report_path=int8_run.report_path,

                    fp16_precision_metric=fp16_run.metrics.precision,
                    int8_precision_metric=int8_run.metrics.precision,
                    precision_loss=_delta(fp16_run.metrics.precision, int8_run.metrics.precision),

                    fp16_recall=fp16_run.metrics.recall,
                    int8_recall=int8_run.metrics.recall,
                    recall_loss=_delta(fp16_run.metrics.recall, int8_run.metrics.recall),

                    fp16_map50=fp16_run.metrics.map50,
                    int8_map50=int8_run.metrics.map50,
                    map50_loss=_delta(fp16_run.metrics.map50, int8_run.metrics.map50),

                    fp16_map50_95=fp16_run.metrics.map50_95,
                    int8_map50_95=int8_run.metrics.map50_95,
                    map50_95_loss=_delta(fp16_run.metrics.map50_95, int8_run.metrics.map50_95),
                )
            )

        return results

    def _build_summary_stats(self, records: List[AccuracyLossRecord]) -> Dict[str, Any]:
        def avg(values: List[Optional[float]]) -> Optional[float]:
            vals = [v for v in values if v is not None]
            if not vals:
                return None
            return sum(vals) / len(vals)

        return {
            "pair_count": len(records),
            "avg_precision_loss": avg([r.precision_loss for r in records]),
            "avg_recall_loss": avg([r.recall_loss for r in records]),
            "avg_map50_loss": avg([r.map50_loss for r in records]),
            "avg_map50_95_loss": avg([r.map50_95_loss for r in records]),
        }

    def write_json(self, output_path: str, records: List[AccuracyLossRecord]) -> None:
        output = {
            "generated_at": datetime.now().isoformat(),
            "summary": self._build_summary_stats(records),
            "records": [asdict(r) for r in records],
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def write_csv(self, output_path: str, records: List[AccuracyLossRecord]) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "pairing_key",
            "model_name",
            "backend_type",
            "dataset_name",
            "input_size",
            "fp16_run_name",
            "int8_run_name",
            "fp16_report_path",
            "int8_report_path",
            "fp16_precision_metric",
            "int8_precision_metric",
            "precision_loss",
            "fp16_recall",
            "int8_recall",
            "recall_loss",
            "fp16_map50",
            "int8_map50",
            "map50_loss",
            "fp16_map50_95",
            "int8_map50_95",
            "map50_95_loss",
        ]

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        return "-" if v is None else f"{v:.6f}"

    def build_markdown_section(self, records: List[AccuracyLossRecord]) -> str:
        summary = self._build_summary_stats(records)

        lines = []
        lines.append("## Precision Sensitivity Analysis")
        lines.append("")
        lines.append("Bu bölüm aynı modelin FP16 ve INT8 koşularını eşleştirerek quantization sonrası doğruluk kaybını raporlar.")
        lines.append("")
        lines.append(f"- Eşleşen FP16/INT8 çift sayısı: **{summary['pair_count']}**")

        if summary["avg_map50_loss"] is not None:
            lines.append(f"- Ortalama mAP@0.5 kaybı: **{summary['avg_map50_loss']:.6f}**")
        if summary["avg_map50_95_loss"] is not None:
            lines.append(f"- Ortalama mAP@0.5:0.95 kaybı: **{summary['avg_map50_95_loss']:.6f}**")
        if summary["avg_precision_loss"] is not None:
            lines.append(f"- Ortalama precision kaybı: **{summary['avg_precision_loss']:.6f}**")
        if summary["avg_recall_loss"] is not None:
            lines.append(f"- Ortalama recall kaybı: **{summary['avg_recall_loss']:.6f}**")

        lines.append("")

        if not records:
            lines.append("_FP16 ve INT8 arasında eşleşen uygun run bulunamadı._")
            lines.append("")
            return "\n".join(lines)

        lines.append("| Model | Backend | Dataset | Input | FP16 mAP50 | INT8 mAP50 | Loss | FP16 mAP50:95 | INT8 mAP50:95 | Loss |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")

        for r in records:
            lines.append(
                f"| {r.model_name or '-'} | {r.backend_type or '-'} | {r.dataset_name or '-'} | {r.input_size or '-'} "
                f"| {self._fmt(r.fp16_map50)} | {self._fmt(r.int8_map50)} | {self._fmt(r.map50_loss)} "
                f"| {self._fmt(r.fp16_map50_95)} | {self._fmt(r.int8_map50_95)} | {self._fmt(r.map50_95_loss)} |"
            )

        lines.append("")
        return "\n".join(lines)

    def run(self, output_json_path: str, output_csv_path: str) -> Dict[str, Any]:
        runs = self.collect_runs()
        records = self.build_accuracy_loss_records(runs)

        self.write_json(output_json_path, records)
        self.write_csv(output_csv_path, records)

        return {
            "run_count": len(runs),
            "pair_count": len(records),
            "output_json_path": output_json_path,
            "output_csv_path": output_csv_path,
            "markdown_section": self.build_markdown_section(records),
        }