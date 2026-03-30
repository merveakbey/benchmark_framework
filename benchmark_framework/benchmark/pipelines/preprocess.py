from dataclasses import dataclass

import cv2
import numpy as np
import torch

from benchmark.schemas.data_models import PreprocessMeta


@dataclass
class PreprocessOutput:
    tensor: torch.Tensor
    meta: PreprocessMeta


class YOLOPreprocessor:
    def __init__(self, config: dict):
        model_cfg = config["model"]
        pipe_cfg = config["pipeline"]["preprocess"]
        backend_cfg = config["backend"]

        self.input_size = model_cfg.get("input_size", [640, 640])
        self.letterbox = pipe_cfg.get("letterbox", True)
        self.normalize = pipe_cfg.get("normalize", True)
        self.color_format = pipe_cfg.get("color_format", "rgb").lower()

        self.backend_type = backend_cfg.get("type", "").lower()
        self.model_format = model_cfg.get("format", "").lower()

        print(
            f"[DEBUG] Preprocessor init -> backend={self.backend_type}, "
            f"format={self.model_format}, input_size={self.input_size}, "
            f"letterbox={self.letterbox}"
        )

    def __call__(self, image):
        original_height, original_width = image.shape[:2]
        target_w, target_h = int(self.input_size[0]), int(self.input_size[1])

        # ONNX için kesin shape garantisi:
        # sabit shape isteyen modellerde letterbox yerine doğrudan resize kullanıyoruz.
        force_exact_resize = (
            self.backend_type in ["onnxruntime", "rknn"]
            or self.model_format in ["onnx", "rknn"]
        )

        if force_exact_resize:
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_height, resized_width = resized.shape[:2]
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            pad_left = 0
            pad_top = 0

        elif self.letterbox:
            resized, scale, pad_left, pad_top = self._letterbox(image, (target_w, target_h))
            resized_height, resized_width = resized.shape[:2]
            scale_x = scale
            scale_y = scale

        else:
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_height, resized_width = resized.shape[:2]
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            pad_left = 0
            pad_top = 0

        if resized_width != target_w or resized_height != target_h:
            raise RuntimeError(
                f"Preprocess exact-shape check failed: got {(resized_height, resized_width)}, "
                f"expected {(target_h, target_w)}"
            )

        if self.color_format == "rgb":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        array = resized.astype(np.float32)

        if self.normalize:
            array = array / 255.0

        array = np.transpose(array, (2, 0, 1))   # HWC -> CHW
        array = np.expand_dims(array, axis=0)    # CHW -> NCHW

        tensor = torch.from_numpy(array)

        print(f"[DEBUG] Preprocess tensor shape: {tuple(tensor.shape)}")

        meta = PreprocessMeta(
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_left=pad_left,
            pad_top=pad_top,
            input_size=[target_w, target_h],
        )

        return PreprocessOutput(tensor=tensor, meta=meta)

    def _letterbox(self, image, new_shape=(640, 640), color=(114, 114, 114)):
        shape = image.shape[:2]  # (h, w)

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        target_w, target_h = int(new_shape[0]), int(new_shape[1])

        r = min(target_w / shape[1], target_h / shape[0])

        new_unpad_w = int(shape[1] * r)
        new_unpad_h = int(shape[0] * r)

        resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        dw = target_w - new_unpad_w
        dh = target_h - new_unpad_h

        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )

        return padded, r, left, top