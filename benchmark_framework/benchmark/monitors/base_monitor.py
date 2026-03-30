from abc import ABC, abstractmethod


class BaseMonitor(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def summarize(self) -> dict:
        pass

    @abstractmethod
    def export_raw(self):
        pass