import langcodes

from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection, MorphynetSubset


class Portuguese_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self):
        super().__init__(language=langcodes.find("Portuguese"))


class Portuguese_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def __init__(self):
        super().__init__(language=langcodes.find("Portuguese"))

    def _getRemoteFilename(self, langcode: str, subset: MorphynetSubset) -> str:
        return "pt.inflectional.v1.tsv"
