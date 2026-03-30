import csv
from pathlib import Path

from benchmark.reporters.base_reporter import BaseReporter


class CSVReporter(BaseReporter):
    def __init__(self, output_dir: str, config: dict, filename: str = "summary.csv"):
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.config = config

    def write(self, report: dict) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        row = self._build_row(report)

        output_path = self.output_dir / self.filename
        self._write_single_csv(output_path, row)

        global_summary_path = self.config.get("run", {}).get("global_summary_path")
        if global_summary_path:
            global_path = Path(global_summary_path)
            global_path.parent.mkdir(parents=True, exist_ok=True)
            self._append_global_csv(global_path, row)

        return str(output_path)

    def _build_row(self, report: dict) -> dict:
        run_metadata = report.get("run_metadata", {})
        evaluation_summary = report.get("evaluation_summary", {})
        latency_summary = report.get("latency_summary", {})
        monitoring_summary = report.get("monitoring_summary", {})
        monitor_0 = monitoring_summary.get("monitor_0", {})

        return {
            "run_name": run_metadata.get("run_name"),
            "task_type": run_metadata.get("task_type"),
            "backend": run_metadata.get("backend"),
            "precision_mode": run_metadata.get("precision"),
            "dataset_type": report.get("dataset_metadata", {}).get("dataset_type"),
            "split": report.get("dataset_metadata", {}).get("split"),
            "num_images": report.get("dataset_metadata", {}).get("num_images"),
            "num_predictions": evaluation_summary.get("num_predictions"),
            "num_ground_truths": evaluation_summary.get("num_ground_truths"),
            "tp": evaluation_summary.get("tp"),
            "fp": evaluation_summary.get("fp"),
            "fn": evaluation_summary.get("fn"),
            "precision": evaluation_summary.get("precision"),
            "recall": evaluation_summary.get("recall"),
            "map_50": evaluation_summary.get("map_50"),
            "map_50_95": evaluation_summary.get("map_50_95"),
            "avg_full_pipeline_ms": latency_summary.get("stage_breakdown_ms", {}).get("full_pipeline", {}).get("mean"),
            "avg_image_read_ms": latency_summary.get("stage_breakdown_ms", {}).get("image_read", {}).get("mean"),
            "avg_preprocess_ms": latency_summary.get("stage_breakdown_ms", {}).get("preprocess", {}).get("mean"),
            "avg_inference_ms": latency_summary.get("stage_breakdown_ms", {}).get("inference", {}).get("mean"),
            "avg_postprocess_ms": latency_summary.get("stage_breakdown_ms", {}).get("postprocess", {}).get("mean"),
            "avg_fps": latency_summary.get("avg_fps"),
            "inference_fps": latency_summary.get("inference_fps"),
            "monitor_count": monitor_0.get("sample_count"),
            "output_dir": run_metadata.get("output_dir"),
        }

    def _write_single_csv(self, path: Path, row: dict):
        fieldnames = list(row.keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    def _append_global_csv(self, path: Path, row: dict):
        fieldnames = list(row.keys())
        file_exists = path.exists()

        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)