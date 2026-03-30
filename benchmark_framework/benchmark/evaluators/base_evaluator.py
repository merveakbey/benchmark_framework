from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def add_sample(self, predictions, ground_truths) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        pass