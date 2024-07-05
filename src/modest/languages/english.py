import langcodes

# MorphyNet
from ..formats.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection


# For MorphyNet, we have language-agnostic dataset loaders, hence we need very little code to build the English dataset.
class English_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self):
        super().__init__(language=langcodes.find("English"))


class English_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def __init__(self):
        super().__init__(language=langcodes.find("English"))
