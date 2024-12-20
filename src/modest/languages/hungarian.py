import langcodes

from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection, MorphynetSubset


class Hungarian_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self):
        super().__init__(language=langcodes.find("Hungarian"))


class Hungarian_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def __init__(self):
        super().__init__(language=langcodes.find("Hungarian"))

    def _getRemoteFilename(self, langcode: str, subset: MorphynetSubset) -> str:
        return "hu.inflectional.segmentation.v1.tsv"
