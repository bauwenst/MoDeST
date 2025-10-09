from ..interfaces.datasets import Languageish
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection, MorphynetSubset


class Hungarian_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def _getLanguage(self) -> Languageish:
        return "Hungarian"


class Hungarian_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def _getLanguage(self) -> Languageish:
        return "Hungarian"

    def _getRemoteFilename(self, langcode: str, subset: MorphynetSubset) -> str:
        return "hu.inflectional.segmentation.v1.tsv"
