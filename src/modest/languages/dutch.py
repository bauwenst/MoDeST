import langcodes

from ..datasets.webcelex import CelexDataset


class Dutch_Celex(CelexDataset):
    def __init__(self):
        super().__init__(language=langcodes.find("Dutch"))
