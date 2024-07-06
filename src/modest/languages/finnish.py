import langcodes

from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection
from ..datasets.morphochallenge2010 import MorphoChallenge2010Dataset


class Finnish_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self):
        super().__init__(language=langcodes.find("Finnish"))


class Finnish_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def __init__(self):
        super().__init__(language=langcodes.find("Finnish"))


class Finnish_MorphoChallenge2010(MorphoChallenge2010Dataset):
    def __init__(self):
        super().__init__(language=langcodes.find("Finnish"))
