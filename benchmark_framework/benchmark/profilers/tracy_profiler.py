from __future__ import annotations

from contextlib import contextmanager

from benchmark.profilers.base_profiler import BaseProfiler
from benchmark.profilers.timer_profiler import TimerProfiler


class TracyProfiler(BaseProfiler):
    def __init__(self):
        self._timer = TimerProfiler()
        self._tracy_enabled = False
        self._tracy = None

        try:
            import TracyClientBindings as tracy  # gerçek modül
            self._tracy = tracy
            self._tracy_enabled = True

            # thread ismini set et (GUI'de görünür)
            self._tracy.thread_name("benchmark_main")

        except Exception:
            self._tracy = None
            self._tracy_enabled = False

    def start_stage(self, stage_name: str) -> None:
        self._timer.start_stage(stage_name)

    def end_stage(self, stage_name: str) -> None:
        self._timer.end_stage(stage_name)

    def record_value(self, key: str, value) -> None:
        self._timer.record_value(key, value)

        # Tracy plot (optional ama çok iyi feature)
        if self._tracy_enabled:
            try:
                self._tracy.plot(key, float(value))
            except Exception:
                pass

    @contextmanager
    def profile_stage(self, stage_name: str):
        zone = None
        self.start_stage(stage_name)

        try:
            if self._tracy_enabled:
                zone = self._enter_zone(stage_name)

            yield

        finally:
            if zone is not None:
                self._exit_zone(zone)

            self.end_stage(stage_name)

    def summarize(self) -> dict:
        summary = self._timer.summarize()

        summary["tracy"] = {
            "enabled": self._tracy_enabled
        }

        return summary

    def reset(self) -> None:
        self._timer.reset()

    def export_raw(self):
        return self._timer.export_raw()

    # ======================
    # 🔥 REAL TRACY ZONE
    # ======================

    def _enter_zone(self, stage_name: str):
        try:
            return self._tracy._ScopedZone(stage_name)
        except Exception:
            return None

    def _exit_zone(self, zone) -> None:
        try:
            # Python binding'de zone otomatik kapanır (GC ile)
            # ama manuel delete etmek daha güvenli
            del zone
        except Exception:
            pass