from typing import Any, List, Optional

import cv2
import numpy as np
import torch

from benchmark.schemas.data_models import DetectionPrediction


class YOLOPostprocessor:
    def __init__(self, config: dict):
        post_cfg = config["pipeline"]["postprocess"]
        self.conf_threshold = post_cfg.get("conf_threshold", 0.25)
        self.iou_threshold = post_cfg.get("iou_threshold", 0.45)
        self.max_det = post_cfg.get("max_det", 300)

    def __call__(self, dataset_item, raw_output, adapter, preprocess_meta) -> List[DetectionPrediction]:
        backend_name = adapter.get_backend_name()

        if backend_name == "rknn":
            return self._decode_rknn_yolo_output(dataset_item, raw_output, adapter, preprocess_meta)

        if backend_name == "tensorrt":
            return self._parse_direct_boxes(dataset_item, raw_output, adapter)

        detections = self._extract_detection_tensor(raw_output)
        if detections is None:
            return []

        if isinstance(detections, torch.Tensor):
            detections = detections.detach().cpu().numpy()

        if not isinstance(detections, np.ndarray):
            return []

        if len(detections.shape) == 3:
            if detections.shape[0] == 0:
                return []
            detections = detections[0]

        if len(detections.shape) != 2:
            return []

        return self._build_predictions_from_direct_boxes(
            dataset_item=dataset_item,
            detections=detections,
            adapter=adapter,
            preprocess_meta=preprocess_meta,
            already_original_coords=False,
        )

    def _decode_rknn_yolo_output(self, dataset_item, raw_output, adapter, preprocess_meta):
        arr = self._extract_detection_tensor(raw_output)
        if arr is None:
            return []

        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()

        if not isinstance(arr, np.ndarray):
            return []

        # Beklenen ham çıktı: (1, 15, 8400)
        if arr.ndim != 3:
            print(f"[DEBUG] RKNN decode skipped: unexpected ndim={arr.ndim}, shape={getattr(arr, 'shape', None)}")
            return []

        # Çalışan eski koddaki mantık:
        # pred = outputs[0][0].T
        # Eğer shape (1, C, N) ise -> (N, C)
        if arr.shape[0] == 1:
            pred = arr[0].T
        else:
            print(f"[DEBUG] RKNN decode skipped: unexpected raw shape={arr.shape}")
            return []

        num_classes = len(adapter.config["model"].get("class_names", []))
        expected_channels = 4 + num_classes  # 4 bbox + nc

        if pred.shape[1] != expected_channels:
            print(f"[DEBUG] RKNN decode skipped: expected channels={expected_channels}, got shape={pred.shape}")
            return []

        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_ids)), class_ids]

        print(
            f"[DEBUG] RKNN score stats -> min={confs.min():.6f}, "
            f"max={confs.max():.6f}, mean={confs.mean():.6f}"
        )

        mask = confs >= self.conf_threshold
        boxes_cxcywh = boxes_cxcywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        print(f"[DEBUG] RKNN kept after conf: {len(confs)}")

        if len(boxes_cxcywh) == 0:
            return []

        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2.0
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2.0
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2.0
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2.0

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        boxes_xyxy = np.nan_to_num(boxes_xyxy, nan=0.0, posinf=0.0, neginf=0.0)

        valid = (
            np.isfinite(boxes_xyxy).all(axis=1) &
            (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) &
            (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        )

        boxes_xyxy = boxes_xyxy[valid]
        confs = confs[valid]
        class_ids = class_ids[valid]

        print(f"[DEBUG] RKNN valid boxes after xyxy conversion: {len(boxes_xyxy)}")

        if len(boxes_xyxy) == 0:
            return []

        # preprocess_meta ile original image'e ölçekleme
        if preprocess_meta is not None:
            scale_x = preprocess_meta.original_width / preprocess_meta.resized_width
            scale_y = preprocess_meta.original_height / preprocess_meta.resized_height
        else:
            scale_x = 1.0
            scale_y = 1.0

        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        orig_w = float(dataset_item.width)
        orig_h = float(dataset_item.height)

        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)

        boxes_for_nms = []
        for box in boxes_xyxy:
            bx1, by1, bx2, by2 = box.tolist()
            boxes_for_nms.append([float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1)])

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            confs.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )

        if len(indices) == 0:
            print("[DEBUG] RKNN kept after NMS: 0")
            return []

        indices = np.array(indices).reshape(-1)
        print(f"[DEBUG] RKNN kept after NMS: {len(indices)}")

        predictions: List[DetectionPrediction] = []
        class_names = adapter.config["model"].get("class_names", [])

        for i in indices[: self.max_det]:
            bx1, by1, bx2, by2 = boxes_xyxy[i].tolist()
            score = float(confs[i])
            class_id = int(class_ids[i])

            class_name = (
                class_names[class_id]
                if 0 <= class_id < len(class_names)
                else str(class_id)
            )

            predictions.append(
                DetectionPrediction(
                    image_id=dataset_item.image_id,
                    class_id=class_id,
                    class_name=class_name,
                    score=score,
                    bbox_xyxy=[float(bx1), float(by1), float(bx2), float(by2)],
                    backend=adapter.get_backend_name(),
                    precision=adapter.get_precision_mode(),
                )
            )

        return predictions

    def _parse_direct_boxes(self, dataset_item, raw_output, adapter):
        detections = self._extract_detection_tensor(raw_output)
        if detections is None:
            return []

        if isinstance(detections, torch.Tensor):
            detections = detections.detach().cpu().numpy()

        if isinstance(detections, np.ndarray) and detections.ndim == 3:
            detections = detections[0]

        if detections is None or len(detections) == 0:
            return []

        return self._build_predictions_from_direct_boxes(
            dataset_item=dataset_item,
            detections=detections,
            adapter=adapter,
            preprocess_meta=None,
            already_original_coords=True,
        )

    def _build_predictions_from_direct_boxes(
        self,
        dataset_item,
        detections: np.ndarray,
        adapter,
        preprocess_meta,
        already_original_coords: bool,
    ) -> List[DetectionPrediction]:
        predictions: List[DetectionPrediction] = []
        class_names = adapter.config["model"].get("class_names", [])

        for det in detections:
            if len(det) < 6:
                continue

            x1, y1, x2, y2, score, class_id = det[:6].tolist()

            if score < self.conf_threshold:
                continue

            if not already_original_coords:
                x1, y1, x2, y2 = self._map_boxes_to_original_image(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    preprocess_meta=preprocess_meta,
                )

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(class_id)
            class_name = (
                class_names[class_id]
                if 0 <= class_id < len(class_names)
                else str(class_id)
            )

            predictions.append(
                DetectionPrediction(
                    image_id=dataset_item.image_id,
                    class_id=class_id,
                    class_name=class_name,
                    score=float(score),
                    bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                    backend=adapter.get_backend_name(),
                    precision=adapter.get_precision_mode(),
                )
            )

            if len(predictions) >= self.max_det:
                break

        return predictions

    def _map_boxes_to_original_image(self, x1, y1, x2, y2, preprocess_meta):
        if preprocess_meta is None:
            return x1, y1, x2, y2

        scale_x = preprocess_meta.scale_x if preprocess_meta.scale_x != 0 else 1.0
        scale_y = preprocess_meta.scale_y if preprocess_meta.scale_y != 0 else 1.0

        x1 = (x1 - preprocess_meta.pad_left) / scale_x
        y1 = (y1 - preprocess_meta.pad_top) / scale_y
        x2 = (x2 - preprocess_meta.pad_left) / scale_x
        y2 = (y2 - preprocess_meta.pad_top) / scale_y

        x1 = max(0.0, min(float(x1), float(preprocess_meta.original_width)))
        y1 = max(0.0, min(float(y1), float(preprocess_meta.original_height)))
        x2 = max(0.0, min(float(x2), float(preprocess_meta.original_width)))
        y2 = max(0.0, min(float(y2), float(preprocess_meta.original_height)))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return x1, y1, x2, y2

    def _extract_detection_tensor(self, raw_output: Any) -> Optional[Any]:
        if raw_output is None:
            return None

        if hasattr(raw_output, "shape"):
            return raw_output

        if isinstance(raw_output, tuple):
            for item in raw_output:
                if hasattr(item, "shape"):
                    return item
            return None

        if isinstance(raw_output, list):
            for item in raw_output:
                if hasattr(item, "shape"):
                    return item
            return None

        return None

    def summarize_raw_output(self, raw_output: Any) -> dict:
        summary = {
            "type": str(type(raw_output))
        }

        if raw_output is None:
            summary["value"] = None
            return summary

        if hasattr(raw_output, "shape"):
            summary["shape"] = list(raw_output.shape)
            return summary

        if isinstance(raw_output, (list, tuple)):
            summary["length"] = len(raw_output)
            items = []

            for item in raw_output:
                if hasattr(item, "shape"):
                    items.append(list(item.shape))
                else:
                    items.append(str(type(item)))

            summary["items"] = items
            return summary

        return summary