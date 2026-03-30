from pathlib import Path
import yaml

from benchmark.core.registry import Registry
from benchmark.reporters.monitor_csv_reporter import MonitorCSVReporter


class BenchmarkRunner:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.registry = Registry(config)

        self.output_dir = None
        self.adapter = None
        self.dataset = None
        self.profiler = None
        self.pipeline = None
        self.evaluator = None
        self.monitors = []
        self.reporters = []

    def setup(self):
        run_name = self.config["run"]["name"]
        self.logger.info(f"Initializing benchmark runner for '{run_name}'")

        self.output_dir = self._build_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory created: {self.output_dir}")

        self.adapter = self.registry.create_adapter()
        self.adapter.load_model()
        self.logger.info(f"Adapter initialized: {self.adapter.get_backend_name()}")

        self.dataset = self.registry.create_dataset()
        self.logger.info(
            f"Dataset initialized: {self.config['dataset']['type']} ({len(self.dataset)} samples)"
        )

        self.profiler = self.registry.create_profiler()
        profiler_type = self.config.get("profiling", {}).get("profiler_type", "timer")
        self.logger.info(f"Profiler initialized: {profiler_type}")

        self.pipeline = self.registry.create_pipeline(self.adapter, self.profiler)
        self.logger.info("Pipeline initialized")

        self.evaluator = self.registry.create_evaluator()
        self.logger.info("Evaluator initialized")

        self.monitors = self.registry.create_monitors()
        self.reporters = self.registry.create_reporters(str(self.output_dir), self.config)
        self.logger.info("Setup completed")

    def run(self):
        max_samples = self.config.get("benchmark", {}).get("max_samples")
        warmup_iters = self.config.get("benchmark", {}).get("warmup_iters", 0)

        dataset_len = len(self.dataset)
        num_samples = min(dataset_len, max_samples) if max_samples is not None else dataset_len

        for monitor in self.monitors:
            monitor.start()

        if num_samples > 0 and warmup_iters > 0:
            self.logger.info(f"Running warmup for {warmup_iters} iterations")
            sample_item = self.dataset[0]

            backend_name = self.adapter.get_backend_name()

            if backend_name == "tensorrt":
                warmup_input = sample_item.image
            elif backend_name == "rknn":
                prep = self.pipeline.preprocessor(sample_item.image)
                warmup_input = prep.tensor.detach().cpu().numpy()
            else:
                prep = self.pipeline.preprocessor(sample_item.image)
                warmup_input = prep.tensor

            for _ in range(warmup_iters):
                self.adapter.warmup(warmup_input)

        all_predictions = []
        total_gt = 0

        self.logger.info(f"Running benchmark on {num_samples} samples")
        for idx in range(num_samples):
            item = self.dataset[idx]
            valid_gt = [ann for ann in item.annotations if getattr(ann, "ignore", 0) == 0]
            total_gt += len(valid_gt)

            predictions = self.pipeline.run_single(item)
            all_predictions.extend(predictions)

            self.evaluator.add_sample(predictions, item.annotations)

        for monitor in self.monitors:
            monitor.stop()

        eval_result = self.evaluator.evaluate()

        report = {
            "run_metadata": {
                "run_name": self.config["run"]["name"],
                "task_type": self.config["task"]["type"],
                "backend": self.adapter.get_backend_name(),
                "precision": self.adapter.get_precision_mode(),
                "model_name": self.config.get("model", {}).get("name"),
                "model_path": self.config.get("model", {}).get("path"),
                "dataset_name": self.config.get("dataset", {}).get("name", self.config.get("dataset", {}).get("type")),
                "input_size": self.config.get("model", {}).get("input_size"),
                "output_dir": str(self.output_dir),
            },
            "dataset_metadata": self.dataset.get_dataset_metadata(),
            "evaluation_summary": {
                "status": "coco_style_evaluation_ready",
                "num_predictions": len(all_predictions),
                "num_ground_truths": total_gt,
                "tp": eval_result["tp"],
                "fp": eval_result["fp"],
                "fn": eval_result["fn"],
                "precision": eval_result["precision"],
                "recall": eval_result["recall"],
                "map_50": eval_result["map_50"],
                "map_50_95": eval_result["map_50_95"],
                "iou_thresholds": eval_result["iou_thresholds"],
            },
            "latency_summary": self.profiler.summarize(),
            "monitoring_summary": self._summarize_monitors(),
            "classwise_metrics": eval_result["classwise_metrics"],
            "artifacts": {},
            "debug_info": self.pipeline.get_debug_info(),
            "config_snapshot": self.config,
        }

        artifact_paths = {}
        for reporter in self.reporters:
            path = reporter.write(report)
            artifact_paths[reporter.__class__.__name__] = path

        monitor_reporter = MonitorCSVReporter(str(self.output_dir))
        for idx, monitor in enumerate(self.monitors):
            monitor_path = monitor_reporter.write(monitor)
            if monitor_path:
                artifact_paths[f"MonitorCSVReporter_{idx}"] = monitor_path

        report["artifacts"] = artifact_paths

        self._write_config_snapshot()

        if getattr(self.pipeline, "visualizer", None) is not None:
            try:
                self.pipeline.visualizer.close()
            except Exception:
                pass

        self.logger.info("Benchmark run completed")
        return report

    def _build_output_dir(self) -> Path:
        output_root = Path(self.config["run"]["output_root"])

        model_name = self.config["model"]["name"]
        backend = self.config["backend"]["type"]
        device = self.config["backend"].get("device", "unknown")
        precision = self.config["model"].get("precision", "unknown")

        folder_name = f"{model_name}_{backend}_{device}_{precision}"
        return output_root / folder_name

    def _write_config_snapshot(self):
        snapshot_path = self.output_dir / "config_snapshot.yaml"
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)

    def _summarize_monitors(self):
        summary = {}
        for idx, monitor in enumerate(self.monitors):
            summary[f"monitor_{idx}"] = monitor.summarize()
        return summary