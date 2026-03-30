from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DetectionAnnotation:
    class_id: int
    class_name: str
    bbox_xyxy: List[float]
    iscrowd: int = 0
    ignore: int = 0


@dataclass
class DatasetItem:
    image_id: str
    image_path: str
    image: Optional[object]
    width: int
    height: int
    annotations: List[DetectionAnnotation]


@dataclass
class DetectionPrediction:
    image_id: str
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: List[float]
    backend: str
    precision: str


@dataclass
class PreprocessMeta:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    input_size: List[int]


@dataclass
class MonitorSample:
    timestamp: float
    cpu_percent: Optional[float] = None
    ram_mb: Optional[float] = None
    temperature_c: Optional[float] = None
    npu_percent: Optional[float] = None