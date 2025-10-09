from ..interfaces.datasets import Languageish
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection
from ..datasets.webcelex import CelexDataset


class German_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def _getLanguage(self) -> Languageish:
        return "German"


class German_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def _getLanguage(self) -> Languageish:
        return "German"


class German_Celex(CelexDataset):
    def _getLanguage(self) -> Languageish:
        return "German"
