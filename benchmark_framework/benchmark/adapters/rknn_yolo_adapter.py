from pathlib import Path
from typing import Any, Dict

import numpy as np


class RKNNYOLOAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = Path(config["model"]["path"])
        self.device = config["backend"].get("device", "rk3588")
        self.precision = config["model"].get("precision", "unknown").lower()
        self.imgsz = config["model"].get("input_size", [640, 640])

        self.rknn = None
        self.runtime_initialized = False

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model file not found: {self.model_path}")

        rknn_cls = None
        import_errors = []

        for module_name in ["rknnlite.api", "rknn_lite.api"]:
            try:
                mod = __import__(module_name, fromlist=["RKNNLite"])
                rknn_cls = getattr(mod, "RKNNLite", None)
                if rknn_cls is not None:
                    print(f"[DEBUG] RKNNLite imported from: {module_name}")
                    break
            except Exception as e:
                import_errors.append(f"{module_name}: {e}")

        if rknn_cls is None:
            raise ImportError(
                "RKNNLite import edilemedi. "
                f"Detaylar: {import_errors}"
            )

        self.rknn = rknn_cls()

        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn başarısız, kod={ret}")

        ret = -1
        init_errors = []

        try:
            ret = self.rknn.init_runtime(target="rk3588")
            print("[DEBUG] RKNN init_runtime(target='rk3588') başarılı")
        except Exception as e:
            init_errors.append(f"init_runtime(target='rk3588') failed: {e}")

        if ret != 0:
            try:
                ret = self.rknn.init_runtime()
                print("[DEBUG] RKNN init_runtime() başarılı")
            except Exception as e:
                init_errors.append(f"init_runtime() failed: {e}")

        if ret != 0:
            raise RuntimeError(
                f"init_runtime başarısız, kod={ret}. Detaylar: {init_errors}"
            )

        self.runtime_initialized = True
        print(f"[DEBUG] RKNN model loaded: {self.model_path}")
        print(f"[DEBUG] RKNN runtime initialized for device: {self.device}")

    def warmup(self, sample_input) -> None:
        if not self.runtime_initialized:
            raise RuntimeError("RKNN runtime is not initialized")
        _ = self.infer(sample_input)

    def infer(self, input_data):
        if not self.runtime_initialized:
            raise RuntimeError("RKNN runtime is not initialized")

        if not isinstance(input_data, np.ndarray):
            raise TypeError(f"RKNN adapter expects np.ndarray, got {type(input_data)}")

        arr = input_data

        # Beklenen giriş: 4D NCHW numpy
        if arr.ndim != 4:
            raise ValueError(f"RKNN adapter expects 4D input, got shape {arr.shape}")

        # NCHW -> NHWC
        if arr.shape[1] in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))

        if arr.dtype != np.float32 and arr.dtype != np.uint8:
            arr = arr.astype(np.float32)

        print(f"[DEBUG] RKNN input shape before inference: {arr.shape}, dtype={arr.dtype}")

        outputs = self.rknn.inference(
            inputs=[arr],
            data_format=["nhwc"],
        )

        if outputs is None:
            print("[DEBUG] RKNN outputs is None")
            return None

        print(f"[DEBUG] RKNN raw outputs type: {type(outputs)}")
        if isinstance(outputs, (list, tuple)):
            print(f"[DEBUG] RKNN raw outputs length: {len(outputs)}")
            for i, out in enumerate(outputs[:5]):
                try:
                    print(f"[DEBUG] RKNN output[{i}] shape: {np.array(out).shape}")
                except Exception:
                    print(f"[DEBUG] RKNN output[{i}] type: {type(out)}")

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def get_backend_name(self) -> str:
        return "rknn"

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
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception:
                pass
        self.rknn = None
        self.runtime_initialized = False