import langcodes

from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection


class Spanish_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self):
        super().__init__(language=langcodes.find("Spanish"))


class Spanish_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def __init__(self):
        super().__init__(language=langcodes.find("Spanish"))
