from abc import ABC, abstractmethod


class BaseReporter(ABC):
    @abstractmethod
    def write(self, report: dict) -> str:
        pass