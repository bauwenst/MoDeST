from typing import Iterable
from abc import ABC, abstractmethod
from pathlib import Path


class ModestDataset(ABC):
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
    def _generate(self, path: Path) -> Iterable:
        """
        Read the given file and generate morphological objects.
        """
        pass

    def generate(self) -> Iterable:
        yield from self._generate(self._get())
