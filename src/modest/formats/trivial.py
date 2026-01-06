from ..interfaces.morphologies import WordDecomposition, WordSegmentation, WordSegmentationWithLemma


class TrivialDecomposition(WordDecomposition):
    """
    Data format that gives you the literal segmentations, ready-made.
    Nothing more, nothing less. Morphemes and morphs separated by a special character.
    """

    def __init__(self, id: int, word: str, decomposition_tag: str, segmentation_tag: str, sep: str):
        super().__init__(id=id, word=word)
        self._morphs    = tuple(segmentation_tag.split(sep))
        self._morphemes = tuple(decomposition_tag.split(sep))

    def decompose(self) -> tuple[str, ...]:
        return self._morphemes

    def segment(self) -> tuple[str, ...]:
        return self._morphs


class TrivialSegmentation(WordSegmentation):
    """
    Same as TrivialDecomposition except without the morphemes, only the morphs.
    """

    def __init__(self, id: int, word: str,
                 segmentation_tag: str, sep: str):
        super().__init__(id=id, word=word)
        self._morphs = tuple(segmentation_tag.split(sep))

    def segment(self) -> tuple[str, ...]:
        return self._morphs


class TrivialSegmentationWithLemma(WordSegmentationWithLemma):

    def __init__(self, id: int, word: str, lemma: str, segmentation_tag: str, sep: str):
        super().__init__(id=id, word=word, lemma=lemma)
        self.morphs = tuple(segmentation_tag.split(sep))

    def segment(self) -> tuple[str, ...]:
        return self.morphs
