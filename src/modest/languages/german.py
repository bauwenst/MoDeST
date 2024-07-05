import langcodes

# MorphyNet
from ..formats.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection


class German_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self):
        super().__init__(language=langcodes.find("German"))


class German_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def __init__(self):
        super().__init__(language=langcodes.find("German"))
