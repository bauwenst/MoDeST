from typing import Iterable
from abc import ABC, abstractmethod
from pathlib import Path


class ModestDataset(ABC):

    @abstractmethod
    def _load(self) -> Path:
        """
        Ensure that the dataset is present.
        """
        pass

    @abstractmethod
    def _generate(self, path: Path) -> Iterable:
        pass

    def generate(self) -> Iterable:
        yield from self._generate(self._load())
