"""
Since word segmentations are often computed on-the-fly, they can have non-negligible impact on speed.
Here we implement a wrapper that precomputes segmentations, stores them to disk, and then returns those without
the original overhead.

We only provide two variants: segmentations and segmentations with lemma. Decompositions are usually precomputed anyway,
so for those the original dataset will do.
"""
from typing import Any, Iterator, TypeVar, Self, TextIO, Generator, Generic
from abc import abstractmethod
from pathlib import Path

from ..interfaces.kernels import ModestKernel, Raw, Writer
from ..interfaces.datasets import ModestDataset, M, Languageish
from ..interfaces.morphologies import WordSegmentation, WordSegmentationWithLemma
from ..formats.trivial import TrivialSegmentation, TrivialSegmentationWithLemma
from ..formats.tsv import iterateTsv
from ..paths import PathManagement


Nested = TypeVar("Nested", bound=WordSegmentation)
class _ModestDatasetPrecomputedBase(ModestDataset[M], Generic[Nested,M]):

    def __init__(self, nested: ModestDataset[Nested]):
        super().__init__()
        self._nested = nested

    def getCollectionName(self) -> str:
        return self._nested.getCollectionName()

    def _getLanguage(self) -> Languageish:
        return self._nested.getLanguage()

    @abstractmethod
    def _kernel(self) -> "_PrecomputedKernel[Nested,M]":
        pass

    def _kernels(self) -> list[ModestKernel[tuple[str,...],M]]:
        return [self._kernel()]

    def _files(self) -> list[Path]:
        folder = PathManagement.namelessCache()
        cache_file = folder / f"precomputed_{self.identifier()}.tsv"
        if not cache_file.exists():
            kernel = self._kernel()
            stream = TsvWriter().openStream(cache_file)
            for obj in self._nested.generate():
                stream.send(kernel._nestedObjectToRaw(obj))
            stream.close()

        return [cache_file]


HasSegmentation = TypeVar("HasSegmentation", bound=WordSegmentation)
class ModestDatasetPrecomputed(_ModestDatasetPrecomputedBase[HasSegmentation,TrivialSegmentation]):
    def _kernel(self):
        return _PrecomputedWithoutLemmaKernel()


HasSegmentationAndLemma = TypeVar("HasSegmentationAndLemma", bound=WordSegmentationWithLemma)
class ModestDatasetPrecomputedWithLemma(_ModestDatasetPrecomputedBase[HasSegmentationAndLemma,TrivialSegmentationWithLemma]):  # Two classes
    def _kernel(self):
        return _PrecomputedWithLemmaKernel()


########################################################################################################################


class _PrecomputedKernel(ModestKernel[tuple[str,...],M], Generic[Nested,M]):

    def _generateRaw(self, path: Path) -> Iterator[tuple[str,...]]:
        yield from iterateTsv(path)

    def _createWriter(self) -> "Writer[tuple[str,...]]":
        return TsvWriter()

    @abstractmethod
    def _nestedObjectToRaw(self, from_nested_dataset: Nested) -> tuple[str, ...]:
        pass


class _PrecomputedWithoutLemmaKernel(_PrecomputedKernel[HasSegmentation,TrivialSegmentation]):

    def _parseRaw(self, raw: tuple[str,...], id: int) -> TrivialSegmentation:
        return TrivialSegmentation(id=id, word=raw[0], segmentation_tag=raw[1], sep=" ")

    def _nestedObjectToRaw(self, from_nested_dataset: HasSegmentation) -> tuple[str, ...]:
        return (from_nested_dataset.word, " ".join(from_nested_dataset.segment()))


class _PrecomputedWithLemmaKernel(_PrecomputedKernel[HasSegmentationAndLemma,TrivialSegmentationWithLemma]):

    def _parseRaw(self, raw: tuple[str,...], id: int) -> TrivialSegmentationWithLemma:
        return TrivialSegmentationWithLemma(id=id, lemma=raw[0], word=raw[1], segmentation_tag=raw[2], sep=" ")

    def _nestedObjectToRaw(self, from_nested_dataset: HasSegmentationAndLemma) -> tuple[str, ...]:
        return (from_nested_dataset.lemma, from_nested_dataset.word, " ".join(from_nested_dataset.segment()))


########################################################################################################################


class TsvWriter(Writer[tuple[str,...]]):
    def _createStream(self, output_path: Path) -> Generator[None, tuple[str,...], None]:
        with open(output_path, "w", encoding="utf-8") as handle:
            while True:
                try:
                    received_tuple = yield
                except GeneratorExit:
                    break
                handle.write("\t".join(map(lambda s: s.replace("\t", " "), received_tuple)) + "\n")
