from pathlib import Path
from typing import Any, Dict

import torch


class PyTorchYOLOAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.model_path = Path(config["model"]["path"])

        requested_device = config["backend"].get("device", "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            requested_device = "cpu"
        self.device = requested_device

        self.precision = config["model"].get("precision", "fp32").lower()
        self._raw_checkpoint = None

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self._raw_checkpoint = checkpoint

        print(f"[DEBUG] Loaded object type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())[:20]}")

            if "ema" in checkpoint and hasattr(checkpoint["ema"], "eval"):
                self.model = checkpoint["ema"]
            elif "model" in checkpoint and hasattr(checkpoint["model"], "eval"):
                self.model = checkpoint["model"]
            else:
                raise TypeError(
                    "Loaded .pt file is a checkpoint dict, but no usable 'ema' or 'model' object was found."
                )
        else:
            self.model = checkpoint

        if not hasattr(self.model, "eval"):
            raise TypeError(f"Loaded object is not a usable PyTorch model. Type: {type(self.model)}")

        self.model.eval()

        if hasattr(self.model, "to"):
            self.model.to(self.device)

        # Kritik nokta:
        # CPU'da fp16 istemiyoruz; modeli zorla float32 yapıyoruz.
        if self.device == "cpu":
            if hasattr(self.model, "float"):
                self.model.float()
            self.precision = "fp32"
        else:
            # CUDA'da istenirse fp16 kullan
            if self.precision == "fp16":
                if hasattr(self.model, "half"):
                    self.model.half()
            else:
                if hasattr(self.model, "float"):
                    self.model.float()

        first_param = next(self.model.parameters(), None)
        if first_param is not None:
            print(f"[DEBUG] Model dtype after load: {first_param.dtype}")
            print(f"[DEBUG] Model device after load: {first_param.device}")

    def warmup(self, sample_input) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        with torch.no_grad():
            _ = self.infer(sample_input)

    def infer(self, input_data):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        if not isinstance(input_data, torch.Tensor):
            raise TypeError("Expected input_data to be a torch.Tensor")

        tensor = input_data.to(self.device)

        # Model dtype ile input dtype aynı olsun
        if self.device == "cpu":
            tensor = tensor.float()
        else:
            if self.precision == "fp16":
                tensor = tensor.half()
            else:
                tensor = tensor.float()

        with torch.no_grad():
            output = self.model(tensor)

        return output

    def get_backend_name(self) -> str:
        return "pytorch"

    def get_precision_mode(self) -> str:
        return self.precision

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "path": str(self.model_path),
            "device": self.device,
            "precision": self.precision,
        }

    def release(self) -> None:
        self.model = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()