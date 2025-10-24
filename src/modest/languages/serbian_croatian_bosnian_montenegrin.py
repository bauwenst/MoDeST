from typing import Iterable, Iterator

import re

from ..interfaces.datasets import Languageish, M
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection, MorphyNetDerivation

CYRILLIC = re.compile(r"[\u0400-\u04FF]")


class SerboCroatian_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self, cyrillic: bool, verbose: bool=False):
        super().__init__(verbose=verbose)
        self._cyrillic = cyrillic

    def _getLanguage(self) -> Languageish:
        return "Serbo-Croatian"

    def _iterators(self) -> Iterator[Iterator[MorphyNetDerivation]]:
        for iterator in super()._iterators():
            yield filter(lambda obj: self._cyrillic == (CYRILLIC.search(obj.word) is not None), iterator)


# https://github.com/kbatsuren/MorphyNet/issues/9
# class SerboCroatian_MorphyNet_Inflections(MorphyNetDataset_Inflection):
#     def _getLanguage(self) -> Languageish:
#         return "Serbo-Croatian"
