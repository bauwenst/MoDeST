from ..interfaces.datasets import Languageish
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection
from ..datasets.morphochallenge2010 import MorphoChallenge2010Dataset


class Finnish_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def _getLanguage(self) -> Languageish:
        return "Finnish"


class Finnish_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def _getLanguage(self) -> Languageish:
        return "Finnish"


class Finnish_MorphoChallenge2010(MorphoChallenge2010Dataset):
    def _getLanguage(self) -> Languageish:
        return "Finnish"
