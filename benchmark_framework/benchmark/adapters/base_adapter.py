from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelAdapter(ABC):
    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self, sample_input: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, input_data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_backend_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_precision_mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_model_metadata(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError
