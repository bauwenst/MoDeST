from typing import Iterable, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import langcodes

from ..paths import PathManagement


@dataclass
class DatasetCard:
    name: str
    language: str
    size: int


M = TypeVar("M")
class ModestDataset(ABC, Generic[M]):
    """
    The responsibilities of this class's descendants are
        1. Knowing how to talk to the external source that carries the data;
        2. Reading those data once they have been pulled and transforming the raw data into the relevant constructor arguments.

    Additionally, because the only user-facing method is .generate() which the subclasses don't implement, you have to
    communicate the type of the objects it generates via a different route, namely with a generic type variable: when
    you extend this abstract class, do it like `class ActualDataset(ModestDataset[TypeOfTheGeneratedObjects]): ...` with
    square brackets inside the inheritance parentheses.
    """

    def __init__(self, name: str, language: langcodes.Language):
        self._name     = name
        self._language = language

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

    def _getCachePath(self) -> Path:
        return PathManagement.datasetCache(language=self._language, dataset_name=self._name)

    def generate(self) -> Iterable[M]:
        yield from self._generate(self._get())

    def card(self) -> DatasetCard:
        return DatasetCard(name=self._name, language=self._language.language_name(), size=count(self.generate()))


def count(it: Iterable) -> int:
    c = 0
    for _ in it:
        c += 1
    return c
