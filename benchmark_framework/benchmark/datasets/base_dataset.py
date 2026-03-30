from abc import ABC, abstractmethod


class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def get_ground_truth(self):
        pass

    @abstractmethod
    def get_dataset_metadata(self) -> dict:
        pass