from ..interfaces.datasets import Languageish
from ..datasets.morphochallenge2010 import MorphoChallenge2010Dataset


class Turkish_MorphoChallenge2010(MorphoChallenge2010Dataset):
    def _getLanguage(self) -> Languageish:
        return "Turkish"
