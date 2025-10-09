from typing import TypeVar, List, Iterator, Self
from pathlib import Path

from tktkt.util.iterables import allEqual

from ..interfaces.datasets import ModestDataset


T = TypeVar("T")

class ChainedModestDatasets(ModestDataset[T]):

    def __init__(self, datasets: List[ModestDataset[T]]):
        assert datasets
        assert allEqual(dataset._language for dataset in datasets)
        super().__init__("+".join(dataset._name for dataset in datasets), datasets[0]._language)

        self._datasets = datasets

    def _get(self) -> List[Path]:
        return [d._get() for d in self._datasets]

    def _generate(self, path: List[Path]) -> Iterator[T]:
        for d in self._datasets:
            yield from d.generate()

    def generate(self) -> Iterator[T]:
        yield from self._generate(self._get())

    def rerouted(self, path: Path) -> Self:
        raise NotImplementedError
