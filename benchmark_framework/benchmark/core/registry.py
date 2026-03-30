from benchmark.adapters.onnx_yolo_adapter import ONNXYOLOAdapter
from benchmark.adapters.pytorch_yolo_adapter import PyTorchYOLOAdapter
from benchmark.adapters.tensorrt_yolo_adapter import TensorRTYOLOAdapter
from benchmark.adapters.rknn_yolo_adapter import RKNNYOLOAdapter

from benchmark.datasets.visdrone_dataset import VisDroneDataset

from benchmark.evaluators.coco_detection_evaluator import COCODetectionEvaluator

from benchmark.monitors.system_monitor import SystemMonitor

from benchmark.pipelines.detection_pipeline import DetectionPipeline
from benchmark.pipelines.visualizer import DetectionVisualizer

from benchmark.profilers.timer_profiler import TimerProfiler
from benchmark.profilers.tracy_profiler import TracyProfiler

from benchmark.reporters.csv_reporter import CSVReporter
from benchmark.reporters.json_reporter import JSONReporter


class Registry:
    def __init__(self, config: dict):
        self.config = config

    def create_adapter(self):
        model_format = self.config["model"]["format"].lower()
        backend_type = self.config["backend"]["type"].lower()

        if model_format == "pt" and backend_type == "pytorch":
            return PyTorchYOLOAdapter(self.config)

        if model_format == "onnx" and backend_type == "onnxruntime":
            return ONNXYOLOAdapter(self.config)

        if model_format == "engine" and backend_type == "tensorrt":
            return TensorRTYOLOAdapter(self.config)

        if model_format == "rknn" and backend_type == "rknn":
            return RKNNYOLOAdapter(self.config)

        raise ValueError(
            f"Unsupported adapter combination: model.format={model_format}, backend.type={backend_type}"
        )

    def create_dataset(self):
        dataset_type = self.config["dataset"]["type"]

        if dataset_type == "visdrone":
            return VisDroneDataset(self.config)

        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def create_profiler(self):
        profiling_cfg = self.config.get("profiling", {})
        profiler_type = profiling_cfg.get("profiler_type", "timer").lower()

        if profiler_type == "timer":
            return TimerProfiler()

        if profiler_type == "tracy":
            return TracyProfiler()

        raise ValueError(f"Unsupported profiler type: {profiler_type}")

    def create_visualizer(self):
        vis_cfg = self.config.get("visualization", {})
        if not vis_cfg.get("enabled", False):
            return None
        return DetectionVisualizer(self.config)

    def create_pipeline(self, adapter, profiler):
        visualizer = self.create_visualizer()
        return DetectionPipeline(
            adapter=adapter,
            profiler=profiler,
            config=self.config,
            visualizer=visualizer,
        )

    def create_evaluator(self):
        iou_thresholds = self.config.get("evaluation", {}).get("iou_thresholds")
        return COCODetectionEvaluator(iou_thresholds=iou_thresholds)

    def create_monitors(self):
        monitors = []
        monitoring_cfg = self.config.get("monitoring", {})

        if not monitoring_cfg.get("enabled", False):
            return monitors

        sample_interval_ms = monitoring_cfg.get("sample_interval_ms", 500)
        thermal_enabled = monitoring_cfg.get("thermal", True)

        if monitoring_cfg.get("system", True):
            monitors.append(
                SystemMonitor(
                    sample_interval_ms=sample_interval_ms,
                    thermal_enabled=thermal_enabled,
                )
            )

        return monitors

    def create_reporters(self, output_dir: str, config: dict):
        reporters = []
        formats = config.get("reporting", {}).get("formats", [])

        if "json" in formats:
            reporters.append(JSONReporter(output_dir))

        if "csv" in formats:
            reporters.append(CSVReporter(output_dir, config))

        return reporters