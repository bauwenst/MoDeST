from typing import Iterable, Iterator, TypeVar, Generic, Union, Any
from typing_extensions import Self
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from langcodes import Language

from .kernels import ModestKernel
from ..paths import PathManagement
from .morphologies import WordSegmentation
from tktkt.util.types import L

@dataclass
class DatasetCard:
    name: str
    language: Language
    size: int


Languageish = Union[Language, str]

M = TypeVar("M", bound=WordSegmentation)
class ModestDataset(ABC, Generic[M]):
    """
    The responsibilities of this class's descendants are
        1. Knowing how to talk to the external source that carries the data (._get());
        2. Knowing which classes can parse those data (._kernels()).

    Additionally, because the only user-facing method that has nothing to do with metadata is .generate() and the
    subclasses don't implement this, you have to communicate the type of the objects it generates via a generic type variable: when
    you extend this abstract class, do it like `class ActualDataset(ModestDataset[TypeOfTheGeneratedObjects]): ...` with
    square brackets inside the inheritance parentheses.
    """

    def __init__(self):  # It is very important that this constructor not have a language parameter. If it did, then dataset families which have extra arguments (e.g. to control verbosity) would have every one of their language datasets repeat those arguments in their constructor.
        self._rerouted: list[Path] = []

    # Implement when defining the collection:

    @abstractmethod
    def getCollectionName(self) -> str:
        pass

    @abstractmethod
    def _kernels(self) -> list[ModestKernel[Any,M]]:
        pass

    @abstractmethod
    def _files(self) -> list[Path]:
        """
        Talk to the external data repository to pull the data for this dataset locally, and return its storage location.
        Exactly one path is required per kernel.
        """
        pass

    def _getCachePath(self) -> Path:  # Will always be used by _files()
        return PathManagement.datasetCache(language=self.getLanguage(), dataset_name=self.getCollectionName())

    # Implement in language-specific subclasses of the collection:

    @abstractmethod
    def _getLanguage(self) -> Languageish:
        pass

    # Pre-implemented user-facing methods

    def generate(self) -> Iterator[M]:
        kernels = self._kernels()
        paths   = self._files() if not self._rerouted else self._rerouted
        assert len(kernels) == len(paths), f"Got {len(kernels)} kernels but {len(paths)} paths." + bool(self._rerouted)*" Note that this dataset was rerouted."
        for kernel, path in zip(kernels,paths):
            yield from kernel.generateObjects(path)

    def getLanguage(self) -> Language:
        return L(self._getLanguage())

    def identifier(self) -> str:
        """A unique name for the dataset."""
        return self.getLanguage().language_name() + "_" + self.getCollectionName()

    def location(self) -> str:
        return self._getCachePath().as_uri()

    def card(self) -> DatasetCard:
        """A summary of the dataset."""
        return DatasetCard(name=self.getCollectionName(), language=self.getLanguage(), size=count(self.generate()))

    def rerouted(self, paths: Union[Path,list[Path]]) -> Self:
        """
        Override the path used by the generator, bypassing ._get().
        Returns itself so you can call it on the same line as the constructor. Not a constructor argument because otherwise
        all ModestDataset subclasses would need to include it too.

        Note: you should NOT be rerouting to a file unless it is basically equivalent to what ._get() would fetch.
              If you're rerouting to a file to go from (dataset1, format) to (dataset2, format), you should just
              create a new ModestDataset. This is also why the rerouting path comes from the user rather than from
              a method implementation, because otherwise it would be identical to ._get().
        """
        self._rerouted = paths if isinstance(paths, list) else [paths]
        return self


def count(it: Iterable) -> int:
    c = 0
    for _ in it:
        c += 1
    return c
