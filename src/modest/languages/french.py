from ..interfaces.datasets import Languageish
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection


class French_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def _getLanguage(self) -> Languageish:
        return "French"


class French_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def _getLanguage(self) -> Languageish:
        return "French"
