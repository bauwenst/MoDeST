from typing import Tuple

from ..interfaces.morphologies import WordDecomposition


class TrivialDecomposition(WordDecomposition):
    """
    Data format that gives you the literal segmentations, ready-made.
    Nothing more, nothing less. Morphemes and morphs separated by a special character.
    """

    def __init__(self, id: int, word: str, decomposition_tag: str, segmentation_tag: str, sep: str):
        super().__init__(id=id, word=word)
        self.morphs    = tuple(segmentation_tag.split(sep))
        self.morphemes = tuple(decomposition_tag.split(sep))

    def decompose(self) -> Tuple[str, ...]:
        return self.morphemes

    def segment(self) -> Tuple[str, ...]:
        return self.morphs