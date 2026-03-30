from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchmarkReport:
    run_metadata: dict[str, Any] = field(default_factory=dict)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    evaluation_summary: dict[str, Any] = field(default_factory=dict)
    latency_summary: dict[str, Any] = field(default_factory=dict)
    monitoring_summary: dict[str, Any] = field(default_factory=dict)
    classwise_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
