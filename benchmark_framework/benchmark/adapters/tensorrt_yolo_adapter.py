from pathlib import Path
from typing import Any, Dict

import numpy as np
from ultralytics import YOLO


class TensorRTYOLOAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = Path(config["model"]["path"])
        self.device = config["backend"].get("device", "cuda")
        self.precision = config["model"].get("precision", "fp32").lower()
        self.imgsz = config["model"].get("input_size", [640, 640])

        self.model = None

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        print(f"[DEBUG] TensorRT engine loaded: {self.model_path}")

    def warmup(self, sample_input) -> None:
        if self.model is None:
            raise RuntimeError("TensorRT model is not loaded")
        _ = self.infer(sample_input)

    def infer(self, input_data):
        if self.model is None:
            raise RuntimeError("TensorRT model is not loaded")

        if not isinstance(input_data, np.ndarray):
            raise TypeError(f"TensorRT adapter expects raw image np.ndarray, got {type(input_data)}")

        results = self.model.predict(
            source=input_data,
            imgsz=self.imgsz[0],
            verbose=False,
            device=0 if self.device.startswith("cuda") else "cpu",
            conf=0.001,
            iou=0.7,
            max_det=300,
        )

        if results:
            r0 = results[0]
            box_count = 0 if r0.boxes is None else len(r0.boxes)
            print(f"[DEBUG] TensorRT Ultralytics result boxes: {box_count}")
            print(f"[DEBUG] TensorRT orig_shape: {getattr(r0, 'orig_shape', None)}")

            if r0.boxes is not None and len(r0.boxes) > 0:
                try:
                    print(f"[DEBUG] First 5 confs: {r0.boxes.conf[:5].detach().cpu().numpy()}")
                    print(f"[DEBUG] First 5 classes: {r0.boxes.cls[:5].detach().cpu().numpy()}")
                except Exception:
                    pass

        return self._results_to_array(results)

    def _results_to_array(self, results):
        if not results:
            return np.zeros((1, 0, 6), dtype=np.float32)

        res0 = results[0]
        if res0.boxes is None or len(res0.boxes) == 0:
            return np.zeros((1, 0, 6), dtype=np.float32)

        xyxy = res0.boxes.xyxy.detach().cpu().numpy()
        conf = res0.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        cls = res0.boxes.cls.detach().cpu().numpy().reshape(-1, 1)

        arr = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
        arr = np.expand_dims(arr, axis=0)

        print(f"[DEBUG] TensorRT output array shape: {arr.shape}")
        return arr

    def get_backend_name(self) -> str:
        return "tensorrt"

    def get_precision_mode(self) -> str:
        return self.precision

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "path": str(self.model_path),
            "device": self.device,
            "precision": self.precision,
            "imgsz": self.imgsz,
        }

    def release(self) -> None:
        self.model = None