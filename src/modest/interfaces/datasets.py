from typing import Iterable, TypeVar, Generic
from abc import ABC, abstractmethod
from pathlib import Path


M = TypeVar("M")


class ModestDataset(ABC, Generic[M]):
    """
    Responsible for
        1. Knowing how to talk to the external source that carries the data;
        2. Reading those data once they have been pulled and transforming the raw data into the relevant constructor arguments.
    """

    @abstractmethod
    def _get(self) -> Path:
        """
        Talk to the external data repository to pull the data locally, and return the file path to it.
        """
        pass

    @abstractmethod
    def _generate(self, path: Path) -> Iterable[M]:
        """
        Read the given file and generate morphological objects.
        """
        pass

    def generate(self) -> Iterable[M]:
        yield from self._generate(self._get())
