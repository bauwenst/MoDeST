from typing import Any, Iterator
from enum import Enum
from pathlib import Path

from tktkt.util.iterables import allEqual, cat

from ..interfaces.datasets import ModestDataset, Languageish, M, M2
from ..interfaces.readers import ModestReader


class ChainedModestDatasets(ModestDataset[M]):

    def __init__(self, datasets: list[ModestDataset[M]]):
        assert datasets
        assert allEqual(dataset.getLanguage() for dataset in datasets)
        self._datasets = datasets
        super().__init__()

    def getCollectionName(self) -> str:
        return "+".join(dataset.getCollectionName() for dataset in self._datasets)

    def _getLanguage(self) -> Languageish:
        return self._datasets[0].getLanguage()

    def _readers(self) -> list[ModestReader[Any,M]]:
        return list(cat(dataset._readers() for dataset in self._datasets))

    def _files(self) -> list[Path]:
        return list(cat(dataset._files() for dataset in self._datasets))

    def _iterators(self) -> Iterator[Iterator[M]]:
        for dataset in self._datasets:
            for iterator in dataset._iterators():
                yield iterator


class OnExhaustion(Enum):
    IGNORE = 1  # Continue without the exhausted iterator.
    REPEAT = 2  # Restart the exhausted iterator from the beginning, until all iterators have reached their end at least once.
    STOP   = 3  # Immediately stop all other iterators.


class InterleavedModestDatasets(ChainedModestDatasets[M]):

    def __init__(self, datasets: list[ModestDataset[M]], exhaustion_strategy: OnExhaustion=OnExhaustion.IGNORE):
        super().__init__(datasets)
        self._if_exhausted = exhaustion_strategy

    def getCollectionName(self) -> str:
        return "¦".join(dataset.getCollectionName() for dataset in self._datasets)

    def _readers(self) -> list[ModestReader]:
        return []

    def _files(self) -> list[Path]:  # Because InterleavedModestDatasets.generate() cycles through  are no separate iterators (that's the whole point)
        return []

    def _iterators(self) -> Iterator[Iterator[M]]:
        yield self._oneGiganticIterator()

    def _oneGiganticIterator(self) -> Iterator[M]:
        iterators          = [dataset.generate() for dataset in self._datasets]
        iterator_completed = [False              for dataset in self._datasets]
        idx = 0
        while True:
            iterator = iterators[idx]
            try:
                item = next(iterator)
                yield item
                idx = (idx + 1) % len(iterators)
            except StopIteration:
                if self._if_exhausted == OnExhaustion.IGNORE:
                    iterators.pop(idx)
                    iterator_completed.pop(idx)
                    if iterators:
                        idx %= len(iterators)
                    else:
                        break
                elif self._if_exhausted == OnExhaustion.REPEAT:
                    iterator_completed[idx] = True
                    if not all(iterator_completed):
                        iterators[idx] = self._datasets[idx].generate()
                    else:
                        break
                elif self._if_exhausted == OnExhaustion.STOP:
                    break
                else:
                    raise NotImplementedError()
