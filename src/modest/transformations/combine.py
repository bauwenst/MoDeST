from typing import List, Any, Iterator
from enum import Enum
from pathlib import Path

from tktkt.util.iterables import allEqual, cat

from ..interfaces.datasets import ModestDataset, Languageish, M
from ..interfaces.kernels import ModestKernel


class ChainedModestDatasets(ModestDataset[M]):
    # The reason why we don't just implement this class's .generate() as `for d in self._dataset: yield from d.generate()` is that concatenating kernels does this automatically AND allows sampling the result to new files.

    def __init__(self, datasets: List[ModestDataset[M]]):
        assert datasets
        assert allEqual(dataset.getLanguage() for dataset in datasets)
        self._datasets = datasets
        super().__init__()

    def getCollectionName(self) -> str:
        return "+".join(dataset.getCollectionName() for dataset in self._datasets)

    def _getLanguage(self) -> Languageish:
        return self._datasets[0].getLanguage()

    def _kernels(self) -> list[ModestKernel[Any,M]]:
        return list(cat(dataset._kernels() for dataset in self._datasets))

    def _files(self) -> List[Path]:
        return list(cat(dataset._files() for dataset in self._datasets))


class OnExhaustion(Enum):
    IGNORE = 1  # Continue without the exhausted iterator.
    REPEAT = 2  # Restart the exhausted iterator from the beginning, until all iterators have reached their end at least once.
    STOP   = 3  # Immediately stop all other iterators.


class InterleavedModestDatasets(ChainedModestDatasets[M]):

    def __init__(self, datasets: List[ModestDataset[M]], exhaustion_strategy: OnExhaustion=OnExhaustion.IGNORE):
        super().__init__(datasets)
        self._if_exhausted = exhaustion_strategy

    def getCollectionName(self) -> str:
        return "¦".join(dataset.getCollectionName() for dataset in self._datasets)

    def generate(self) -> Iterator[M]:
        pairs = list(self._getKernelsWithFiles())
        iterators          = [kernel.generateObjects(path) for kernel, path in pairs]
        iterator_completed = [False]*len(pairs)
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
                        iterators[idx] = pairs[idx][0].generateObjects(pairs[idx][1])
                        idx = (idx + 1) % len(iterators)
                    else:
                        break
                elif self._if_exhausted == OnExhaustion.STOP:
                    break
                else:
                    raise NotImplementedError()
