from typing import Iterable, Iterator, TypeVar, Generic, Union
from typing_extensions import Self
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


Languageish = Union[langcodes.Language, str]

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

    def __init__(self, name: str, language: Languageish):
        self._name     = name
        self._language = langcodes.find(language) if isinstance(language, str) else language

        self._rerouted: Path = None

    @abstractmethod
    def _get(self) -> Path:
        """
        Talk to the external data repository to pull the data locally, and return the file path to it.
        """
        pass

    @abstractmethod
    def _generate(self, path: Path, **kwargs) -> Iterator[M]:
        """
        Read the given file and generate morphological objects.
        """
        pass

    def _getCachePath(self) -> Path:
        return PathManagement.datasetCache(language=self._language, dataset_name=self._name)

    def generate(self, **kwargs) -> Iterator[M]:
        yield from self._generate(self._get() if not self._rerouted else self._rerouted, **kwargs)

    def identifier(self) -> str:
        return self._name + "_" + self._language.language_name()

    def card(self) -> DatasetCard:
        return DatasetCard(name=self._name, language=self._language.language_name(), size=count(self.generate()))

    def rerouted(self, path: Path) -> Self:
        """
        Override the path used by the generator, bypassing ._get().
        Returns itself so you can call it on the same line as the constructor. Not a constructor argument because otherwise
        all ModestDataset subclasses would need to include it too.

        Note: you should NOT be rerouting to a file unless it is basically equivalent to what ._get() would fetch.
              If you're rerouting to a file to go from (dataset1, format) to (dataset2, format), you should just
              create a new ModestDataset. This is also why there is no way to impute the rerouted path, because that
              would come down to making a new class with a dedicated ._get().
        """
        self._rerouted = path
        return self


def count(it: Iterable) -> int:
    c = 0
    for _ in it:
        c += 1
    return c
