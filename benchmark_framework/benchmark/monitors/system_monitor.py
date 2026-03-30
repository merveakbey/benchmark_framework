import csv
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from benchmark.monitors.base_monitor import BaseMonitor


class SystemMonitor(BaseMonitor):
    def __init__(self, sample_interval_ms: int = 500, thermal_enabled: bool = True):
        self.sample_interval_ms = sample_interval_ms
        self.thermal_enabled = thermal_enabled

        self._samples: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return

        with self._lock:
            self._samples = []

        self._running = True

        psutil.cpu_percent(interval=None)

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._append_sample()

    def summarize(self) -> dict:
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {
                "sample_count": 0,
                "sampling_interval_ms": self.sample_interval_ms,
            }

        cpu_vals = [s["cpu_percent"] for s in samples if s["cpu_percent"] is not None]
        process_ram_vals = [s["process_ram_mb"] for s in samples if s["process_ram_mb"] is not None]
        system_ram_used_vals = [s["system_ram_used_mb"] for s in samples if s["system_ram_used_mb"] is not None]
        system_ram_percent_vals = [s["system_ram_percent"] for s in samples if s["system_ram_percent"] is not None]
        temp_vals = [s["temperature_c"] for s in samples if s["temperature_c"] is not None]

        summary = {
            "sample_count": len(samples),
            "sampling_interval_ms": self.sample_interval_ms,
        }

        if cpu_vals:
            summary["cpu_percent"] = self._build_stats(cpu_vals)

        if process_ram_vals:
            summary["process_ram_mb"] = self._build_stats(process_ram_vals)

        if system_ram_used_vals:
            summary["system_ram_used_mb"] = self._build_stats(system_ram_used_vals)

        if system_ram_percent_vals:
            summary["system_ram_percent"] = self._build_stats(system_ram_percent_vals)

        if temp_vals:
            summary["temperature_c"] = self._build_stats(temp_vals)

        return summary

    def export_raw(self):
        with self._lock:
            return list(self._samples)

    def export_csv(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "datetime_readable",
            "cpu_percent",
            "process_ram_mb",
            "system_ram_used_mb",
            "system_ram_percent",
            "temperature_c",
        ]

        with self._lock:
            rows = list(self._samples)

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return str(path)

    def _run_loop(self):
        interval_sec = max(self.sample_interval_ms / 1000.0, 0.05)

        self._append_sample()

        while self._running:
            time.sleep(interval_sec)

            if not self._running:
                break

            self._append_sample()

    def _append_sample(self) -> None:
        ts = time.time()

        sample = {
            "datetime_readable": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "cpu_percent": self._read_cpu_percent(),
            "process_ram_mb": self._read_process_ram_mb(),
            "system_ram_used_mb": self._read_system_ram_used_mb(),
            "system_ram_percent": self._read_system_ram_percent(),
            "temperature_c": self._read_temperature() if self.thermal_enabled else None,
        }

        with self._lock:
            self._samples.append(sample)

    def _read_cpu_percent(self):
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    def _read_process_ram_mb(self):
        try:
            rss = self._process.memory_info().rss
            return float(rss / (1024 * 1024))
        except Exception:
            return None

    def _read_system_ram_used_mb(self):
        try:
            mem = psutil.virtual_memory()
            used = mem.used / (1024 * 1024)
            return float(used)
        except Exception:
            return None

    def _read_system_ram_percent(self):
        try:
            mem = psutil.virtual_memory()
            return float(mem.percent)
        except Exception:
            return None

    def _read_temperature(self):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                if "k10temp" in temps:
                    for entry in temps["k10temp"]:
                        current = getattr(entry, "current", None)
                        if current is None:
                            continue
                        current = float(current)
                        if 0.0 < current < 150.0:
                            return current

                if "coretemp" in temps:
                    for entry in temps["coretemp"]:
                        current = getattr(entry, "current", None)
                        if current is None:
                            continue
                        current = float(current)
                        if 0.0 < current < 150.0:
                            return current

                if "amdgpu" in temps:
                    for entry in temps["amdgpu"]:
                        current = getattr(entry, "current", None)
                        if current is None:
                            continue
                        current = float(current)
                        if 0.0 < current < 150.0:
                            return current

                if "nvme" in temps:
                    for entry in temps["nvme"]:
                        current = getattr(entry, "current", None)
                        if current is None:
                            continue
                        current = float(current)
                        if 0.0 < current < 150.0:
                            return current

                for _, entries in temps.items():
                    for entry in entries:
                        current = getattr(entry, "current", None)
                        if current is None:
                            continue
                        current = float(current)
                        if 0.0 < current < 150.0:
                            return current
        except Exception:
            pass

        thermal_base = Path("/sys/class/thermal")
        if thermal_base.exists():
            zones = sorted(thermal_base.glob("thermal_zone*/temp"))
            for zone in zones:
                try:
                    raw = zone.read_text().strip()
                    if not raw:
                        continue

                    value = float(raw)
                    if value > 1000.0:
                        value = value / 1000.0

                    if 0.0 < value < 150.0:
                        return value
                except Exception:
                    continue

        return None

    def _build_stats(self, values):
        return {
            "mean": round(statistics.mean(values), 4),
            "min": round(min(values), 4),
            "peak": round(max(values), 4),
        }