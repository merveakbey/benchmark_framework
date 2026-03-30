from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort


class ONNXYOLOAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = Path(config["model"]["path"])
        self.device = config["backend"].get("device", "cpu").lower()

        self.session = None
        self.input_name = None
        self.output_names = None
        self.precision = config["model"].get("precision", "fp32").lower()

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        providers = self._resolve_providers()

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if not inputs:
            raise RuntimeError("ONNX model has no inputs")
        if not outputs:
            raise RuntimeError("ONNX model has no outputs")

        self.input_name = inputs[0].name
        self.output_names = [o.name for o in outputs]

        print(f"[DEBUG] ONNX providers: {self.session.get_providers()}")
        print(f"[DEBUG] ONNX input name: {self.input_name}")
        print(f"[DEBUG] ONNX output names: {self.output_names}")

        for idx, inp in enumerate(inputs):
            print(f"[DEBUG] Input[{idx}] name={inp.name}, shape={inp.shape}, type={inp.type}")

        for idx, out in enumerate(outputs):
            print(f"[DEBUG] Output[{idx}] name={out.name}, shape={out.shape}, type={out.type}")

    def warmup(self, sample_input) -> None:
        if self.session is None:
            raise RuntimeError("ONNX session is not loaded")
        _ = self.infer(sample_input)

    def infer(self, input_data):
        if self.session is None:
            raise RuntimeError("ONNX session is not loaded")

        if hasattr(input_data, "detach"):
            array = input_data.detach().cpu().numpy()
        else:
            array = np.asarray(input_data)

        if self.precision == "fp16":
            array = array.astype(np.float16)
        else:
            array = array.astype(np.float32)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: array},
        )

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def get_backend_name(self) -> str:
        return "onnxruntime"

    def get_precision_mode(self) -> str:
        return self.precision

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "path": str(self.model_path),
            "device": self.device,
            "precision": self.precision,
            "providers": self.session.get_providers() if self.session else [],
        }

    def release(self) -> None:
        self.session = None

    def _resolve_providers(self):
        available = ort.get_available_providers()

        if self.device == "cuda":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("[WARN] CUDA requested for ONNX Runtime but CUDAExecutionProvider not available. Falling back to CPU.")

        return ["CPUExecutionProvider"]