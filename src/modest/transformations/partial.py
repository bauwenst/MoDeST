"""
Selects only one/some of the splits in each example of a dataset.
"""
from typing import Iterator, Generic, TypeVar, Any
from pathlib import Path
from abc import abstractmethod

from ..interfaces.datasets import ModestDataset, M, Languageish
from ..interfaces.kernels import ModestKernel, Raw
from ..interfaces.morphologies import WordSegmentation


class SingleSplitSegmentation(WordSegmentation):

    def __init__(self, nested: WordSegmentation, last_not_first: bool):
        super().__init__(nested._id, nested.word)
        self._nested = nested
        self._last_not_first = last_not_first

    def segment(self) -> tuple[str, ...]:
        morphs = self._nested.segment()
        if self._last_not_first:
            first = "".join(morphs[:-1])
            return (first, morphs[-1]) if first else morphs
        else:
            last = "".join(morphs[1:])
            return (morphs[0], last) if last else morphs

    def __getattr__(self, item):
        # Only called on an error.
        return getattr(self._nested, item)


class AllButFirstSplitSegmentation(WordSegmentation):

    def __init__(self, nested: WordSegmentation):
        super().__init__(nested._id, nested.word)
        self._nested = nested

    def segment(self) -> tuple[str, ...]:
        morphs = self._nested.segment()
        if len(morphs) > 2:  # stem + suffix1 + suffix2
            return (morphs[0] + morphs[1],) + morphs[2:]
        else:
            return morphs


########################################################################################################################

M2 = TypeVar("M2", bound=WordSegmentation)

class _ConvertedSegmentationsDataset(ModestDataset[M2], Generic[M,M2]):

    def __init__(self, wrapper_name: str, nested_dataset: ModestDataset[M]):
        super().__init__()
        self._nested_dataset = nested_dataset
        self._wrapper_name   = wrapper_name

    def getCollectionName(self) -> str:
        return self._nested_dataset.getCollectionName() + "-" + self._wrapper_name

    def _getLanguage(self) -> Languageish:
        return self._nested_dataset._getLanguage()

    def _kernels(self) -> list[ModestKernel[Any,M]]:
        return self._nested_dataset._kernels()

    def _files(self) -> list[Path]:
        return self._nested_dataset._files()

    def generate(self) -> Iterator[M2]:
        for segmentation in super().generate():
            yield self._convertSegmentation(segmentation)

    @abstractmethod
    def _convertSegmentation(self, original_segmentation: M) -> M2:
        pass


class SingleSplitDataset(_ConvertedSegmentationsDataset[M,SingleSplitSegmentation]):
    """
    MoDeST dataset that is reinterpreted by only considering one split.
    This makes it equivalent to MorphScore.
    """

    def __init__(self, nested_dataset: ModestDataset[M], last_not_first: bool=False, n_morphs_minimum: int=1):
        super().__init__(wrapper_name="OnlyFirst", nested_dataset=nested_dataset, n_morphs_minimum=n_morphs_minimum)
        self._last_not_first = last_not_first

    def _convertSegmentation(self, original_segmentation: M) -> SingleSplitSegmentation:
        return SingleSplitSegmentation(original_segmentation, self._last_not_first)


class AllButFirstSplitDataset(_ConvertedSegmentationsDataset[M,AllButFirstSplitSegmentation]):
    """
    MoDeST dataset that is reinterpreted by considering all splits but the first.
    """

    def __init__(self, nested_dataset: ModestDataset[M], n_morphs_minimum: int=3):
        super().__init__(wrapper_name="NotFirst", nested_dataset=nested_dataset, n_morphs_minimum=n_morphs_minimum)

    def _convertSegmentation(self, original_segmentation: M) -> AllButFirstSplitSegmentation:
        return AllButFirstSplitSegmentation(original_segmentation)
