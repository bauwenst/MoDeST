from typing import Iterable, Iterator, TypeVar, Generic, Union
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from langcodes import Language
import langcodes

from ..paths import PathManagement
from .morphologies import WordSegmentation
from tktkt.util.types import L

@dataclass
class DatasetCard:
    name: str
    language: str
    size: int


Languageish = Union[Language, str]

M = TypeVar("M", bound=WordSegmentation)
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

    def __init__(self):  # It is very important that this constructor not have a language parameter. If it did, then dataset families which have extra arguments (e.g. to control verbosity) would have every one of their language datasets repeat those arguments in their constructor.
        self._rerouted: Path = None

    @abstractmethod
    def getName(self) -> str:
        pass

    @abstractmethod
    def _getLanguage(self) -> Languageish:
        pass

    def getLanguage(self) -> Language:
        return L(self._getLanguage())

    @abstractmethod
    def _get(self) -> Path:
        """
        Talk to the external data repository to pull the data locally, and return the file path to it.
        """
        pass

    @abstractmethod
    def _generate(self, path: Path) -> Iterator[M]:
        """
        Read the given file and generate morphological objects.
        """
        pass

    def _getCachePath(self) -> Path:
        return PathManagement.datasetCache(language=self.getLanguage(), dataset_name=self.getName())

    def generate(self) -> Iterator[M]:
        yield from self._generate(self._get() if not self._rerouted else self._rerouted)

    def identifier(self) -> str:
        return self.getName() + "_" + self.getLanguage().language_name()

    def card(self) -> DatasetCard:
        return DatasetCard(name=self.getName(), language=self.getLanguage().language_name(), size=count(self.generate()))

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
