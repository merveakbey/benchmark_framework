from abc import ABC, abstractmethod
from contextlib import contextmanager


class BaseProfiler(ABC):
    @abstractmethod
    def start_stage(self, stage_name: str) -> None:
        pass

    @abstractmethod
    def end_stage(self, stage_name: str) -> None:
        pass

    @abstractmethod
    def record_value(self, key: str, value) -> None:
        pass

    @abstractmethod
    def summarize(self) -> dict:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def export_raw(self):
        pass

    @contextmanager
    def profile_stage(self, stage_name: str):
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_stage(stage_name)