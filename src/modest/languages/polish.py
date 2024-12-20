import langcodes

from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection


class Polish_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self):
        super().__init__(language=langcodes.find("Polish"))


# https://github.com/kbatsuren/MorphyNet/issues/9
# class Polish_MorphyNet_Inflections(MorphyNetDataset_Inflection):
#     def __init__(self):
#         super().__init__(language=langcodes.find("Polish"))
