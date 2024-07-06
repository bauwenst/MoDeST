import langcodes

from ..datasets.morphochallenge2010 import MorphoChallenge2010Dataset


class Turkish_MorphoChallenge2010(MorphoChallenge2010Dataset):
    def __init__(self):
        super().__init__(language=langcodes.find("Turkish"))
