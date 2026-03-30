from pathlib import Path
from typing import List
import re

import cv2

from benchmark.schemas.data_models import DatasetItem, DetectionAnnotation


class VisDroneDataset:
    def __init__(self, config: dict):
        self.config = config
        dataset_cfg = config["dataset"]
        model_cfg = config["model"]

        self.root_dir = Path(dataset_cfg["root_dir"])
        self.split = dataset_cfg.get("split", "val")
        self.image_dir = Path(dataset_cfg["image_dir"])
        self.annotation_dir = Path(dataset_cfg["annotation_dir"])
        self.class_names = model_cfg.get("class_names", [])

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")

        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        )

        if not self.image_paths:
            raise RuntimeError(f"No images found in: {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DatasetItem:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Image could not be loaded: {image_path}")

        height, width = image.shape[:2]
        image_id = image_path.stem
        annotation_path = self.annotation_dir / f"{image_id}.txt"
        annotations = self._load_annotations(annotation_path)

        return DatasetItem(
            image_id=image_id,
            image_path=str(image_path),
            image=image,
            width=width,
            height=height,
            annotations=annotations,
        )

    def _load_annotations(self, annotation_path: Path) -> List[DetectionAnnotation]:
        annotations: List[DetectionAnnotation] = []

        if not annotation_path.exists():
            return annotations

        with open(annotation_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [line.rstrip("\n\r") for line in f.readlines()]

        lines = [line.strip().lstrip("\ufeff") for line in raw_lines if line.strip()]

        print(f"[DEBUG] Reading annotation file: {annotation_path.name}, line_count={len(lines)}")

        for line_idx, line in enumerate(lines):
            parts = [p for p in re.split(r"[\s,]+", line.strip()) if p]

            if line_idx < 3:
                print(f"[DEBUG] Raw line {line_idx}: {repr(line)}")
                print(f"[DEBUG] Parsed parts {line_idx}: {parts}")

            if len(parts) < 8:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                score = int(float(parts[4]))
                category = int(float(parts[5]))
                truncation = int(float(parts[6]))
                occlusion = int(float(parts[7]))
            except Exception:
                continue

            if w <= 0 or h <= 0:
                continue

            ignore = 1 if category == 0 else 0
            class_id = category - 1 if category > 0 else -1

            class_name = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else "ignored"
            )

            annotations.append(
                DetectionAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    bbox_xyxy=[x, y, x + w, y + h],
                    ignore=ignore,
                )
            )

        print(f"[DEBUG] Parsed annotations count: {len(annotations)}")
        return annotations

    def get_dataset_metadata(self) -> dict:
        return {
            "dataset_type": "visdrone",
            "split": self.split,
            "num_images": len(self.image_paths),
            "num_classes": len(self.class_names),
            "image_dir": str(self.image_dir),
            "annotation_dir": str(self.annotation_dir),
        }