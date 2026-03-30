import time
from collections import defaultdict

from benchmark.profilers.base_profiler import BaseProfiler


class TimerProfiler(BaseProfiler):
    def __init__(self):
        self._stage_start_times = {}
        self._stage_records = defaultdict(list)
        self._extra_values = defaultdict(list)

    def start_stage(self, stage_name: str) -> None:
        self._stage_start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> None:
        if stage_name not in self._stage_start_times:
            raise KeyError(f"Stage '{stage_name}' was not started")

        elapsed_ms = (time.perf_counter() - self._stage_start_times[stage_name]) * 1000.0
        self._stage_records[stage_name].append(elapsed_ms)
        del self._stage_start_times[stage_name]

    def record_value(self, key: str, value) -> None:
        self._extra_values[key].append(value)

    def summarize(self) -> dict:
        summary = {
            "stage_breakdown_ms": {},
            "extra_values": {},
        }

        for stage_name, values in self._stage_records.items():
            if not values:
                continue

            stage_summary = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }
            summary["stage_breakdown_ms"][stage_name] = stage_summary

        for key, values in self._extra_values.items():
            if not values:
                continue

            value_summary = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }
            summary["extra_values"][key] = value_summary

        inference_mean = summary["stage_breakdown_ms"].get("inference", {}).get("mean")
        full_pipeline_mean = summary["stage_breakdown_ms"].get("full_pipeline", {}).get("mean")

        if inference_mean is not None and inference_mean > 0:
            summary["inference_fps"] = round(1000.0 / inference_mean, 4)

        if full_pipeline_mean is not None and full_pipeline_mean > 0:
            summary["avg_fps"] = round(1000.0 / full_pipeline_mean, 4)

        return summary

    def reset(self) -> None:
        self._stage_start_times.clear()
        self._stage_records.clear()
        self._extra_values.clear()

    def export_raw(self):
        return {
            "stage_records_ms": {k: list(v) for k, v in self._stage_records.items()},
            "extra_values": {k: list(v) for k, v in self._extra_values.items()},
        }