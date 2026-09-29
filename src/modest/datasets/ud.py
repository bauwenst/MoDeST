"""
For segmentations derived from Universal Dependencies (UD), which sometimes offers morphological splits.
"""
import re
from abc import abstractmethod
from typing import Iterator
from pathlib import Path

from datasets import load_dataset, DatasetDict

from ..formats.trivial import TrivialSegmentation
from ..formats.tsv import iterateTsv
from ..interfaces.datasets import ModestDataset, M2
from ..interfaces.readers import ModestReader, Raw

SegmentationRecord = tuple[str,tuple[str, ...]]


class UniversalDependencies(ModestDataset[TrivialSegmentation]):

    def __init__(self, include_monomorphemic: bool=False):
        super().__init__()
        self._include_monomorphemic = include_monomorphemic

    def getCollectionName(self) -> str:
        return "UD"

    @abstractmethod
    def _subset(self) -> str:
        pass

    def _readers(self) -> list[ModestReader[SegmentationRecord,TrivialSegmentation]]:
        return [_UDReader(include_monomorphemic=self._include_monomorphemic)]

    def _loadHuggingfaceDataset(self) -> DatasetDict:
        return load_dataset("universal-dependencies/universal_dependencies", self._subset())

    def _files(self) -> list[Path]:
        cache_path = self._getCachePath() / "ud-train.tsv"
        if not cache_path.exists():
            self._datasetToFiles(self._loadHuggingfaceDataset(), cache_path.parent, output_stem="ud")
            assert cache_path.exists()

        return [cache_path]

    @abstractmethod
    def _datasetToFiles(self, input_dataset: DatasetDict, output_folder: Path, output_stem: str= "ud") -> list[Path]:
        pass


class _UDReader(ModestReader[SegmentationRecord,TrivialSegmentation]):
    """
    Reader whose subclasses turn a HF DatasetDict into a stream of (word, morphs) tuples.
    """

    def __init__(self, include_monomorphemic: bool=False):
        super().__init__()
        self._include_monomorphemic = include_monomorphemic

    def _parseRaw(self, raw: SegmentationRecord, id: int) -> TrivialSegmentation:
        return TrivialSegmentation(id=id, word=raw[0], segmentation=raw[1])

    def _generateRaw(self, path: Path) -> Iterator[SegmentationRecord]:
        for row in iterateTsv(path):
            word, parts = row
            parts = tuple(parts.split(" "))
            if self._include_monomorphemic or len(parts) > 1:
                yield word, parts

    def _createWriter(self):
        raise NotImplementedError()
