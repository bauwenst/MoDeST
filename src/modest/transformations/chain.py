from typing import List, Any
from pathlib import Path

from tktkt.util.iterables import allEqual, cat

from ..interfaces.datasets import ModestDataset, Languageish, M
from ..interfaces.kernels import ModestKernel


class ChainedModestDatasets(ModestDataset[M]):

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
