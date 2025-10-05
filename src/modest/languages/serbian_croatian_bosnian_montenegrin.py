from typing import Iterable
from pathlib import Path

import langcodes
import re

from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection, MorphyNetDerivation

CYRILLIC = re.compile(r"[\u0400-\u04FF]")


class SerboCroatian_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self, cyrillic: bool):
        super().__init__(language=langcodes.find("Serbo-Croatian"))
        self._cyrillic = cyrillic

    def _generate(self, path: Path, verbose: bool=False, **kwargs) -> Iterable[MorphyNetDerivation]:
        for obj in super()._generate(path, verbose=verbose, **kwargs):
            is_cyrillic = CYRILLIC.search(obj.word) is not None
            if self._cyrillic and is_cyrillic:
                yield obj
            elif not self._cyrillic and not is_cyrillic:
                yield obj


# https://github.com/kbatsuren/MorphyNet/issues/9
# class SerboCroatian_MorphyNet_Inflections(MorphyNetDataset_Inflection):
#     def __init__(self):
#         super().__init__(language=langcodes.find("Serbo-Croatian"))
