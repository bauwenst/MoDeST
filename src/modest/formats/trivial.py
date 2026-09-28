from typing import Iterable, Union

from ..interfaces.morphologies import WordDecomposition, WordSegmentation, WordSegmentationWithLemma


SequenceOfStrings = Union[list[str], tuple[str,...]]  # Excludes str itself, unlike Sequence[str]


class TrivialDecomposition(WordDecomposition):
    """
    Data format that gives you the literal segmentations, ready-made.
    Nothing more, nothing less. Morphemes and morphs separated by a special character.
    """

    def __init__(self, id: int, word: str, decomposition: SequenceOfStrings, segmentation: SequenceOfStrings):
        super().__init__(id=id, word=word)
        self._morphs    = tuple(segmentation)
        self._morphemes = tuple(decomposition)

    def decompose(self) -> tuple[str, ...]:
        return self._morphemes

    def segment(self) -> tuple[str, ...]:
        return self._morphs


class TrivialSegmentation(WordSegmentation):
    """
    Same as TrivialDecomposition except without the morphemes, only the morphs.
    """

    def __init__(self, id: int, word: str, segmentation: SequenceOfStrings):
        super().__init__(id=id, word=word)
        self._morphs = tuple(segmentation)

    def segment(self) -> tuple[str, ...]:
        return self._morphs


class TrivialSegmentationWithLemma(WordSegmentationWithLemma):

    def __init__(self, id: int, word: str, segmentation: SequenceOfStrings, lemma: str):
        super().__init__(id=id, word=word, lemma=lemma)
        self._morphs = tuple(segmentation)

    def segment(self) -> tuple[str, ...]:
        return self._morphs
