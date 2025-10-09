from ..interfaces.datasets import Languageish
from ..datasets.webcelex import CelexDataset


class Dutch_Celex(CelexDataset):
    def _getLanguage(self) -> Languageish:
        return "Dutch"
