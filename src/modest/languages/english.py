from ..interfaces.datasets import Languageish
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection
from ..datasets.webcelex import CelexDataset
from ..datasets.morphochallenge2010 import MorphoChallenge2010Dataset


class English_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def _getLanguage(self) -> Languageish:
        return "English"


class English_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def _getLanguage(self) -> Languageish:
        return "English"


class English_Celex(CelexDataset):
    def _getLanguage(self) -> Languageish:
        return "English"


class English_MorphoChallenge2010(MorphoChallenge2010Dataset):
    def _getLanguage(self) -> Languageish:
        return "English"
