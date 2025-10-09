from ..interfaces.datasets import Languageish
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection, MorphynetSubset


class Portuguese_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def _getLanguage(self) -> Languageish:
        return "Portuguese"


class Portuguese_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def _getLanguage(self) -> Languageish:
        return "Portuguese"

    def _getRemoteFilename(self, langcode: str, subset: MorphynetSubset) -> str:
        return "pt.inflectional.v1.tsv"
