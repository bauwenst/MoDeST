from typing import Tuple

from tktkt.preparation.mappers import MorphoChallengeCapitals
from ..interfaces.morphologies import WordDecomposition


class MorphoChallenge2010Morphology(WordDecomposition):

    def __init__(self, word: str, segmentation_tag: str, turkish_to_utf8: bool=True):
        if turkish_to_utf8:
            mapping = MorphoChallengeCapitals()  # Only for Turkish, but since it works with capitals and MC2010 lowercases everything else, this isn't a problem.
            word             = mapping.invert(word)
            segmentation_tag = mapping.invert(segmentation_tag)

        super().__init__(word=word)

        self.morph_sequences    = []
        self.morpheme_sequences = []

        for structure in segmentation_tag.split(","):
            morphs = []
            morphemes = []
            for mm in structure.strip().split(" "):
                morph, morpheme = mm.split(":")
                morphs.append(morph)
                morphemes.append(morpheme)  # TODO: Possibly have to call       morpheme, tag = morpheme.split("_")

            self.morph_sequences.append(morphs)
            self.morpheme_sequences.append(morphemes)

    def segment(self) -> Tuple[str, ...]:
        return tuple(self.morph_sequences[0])

    def decompose(self) -> Tuple[str, ...]:
        return tuple(self.morpheme_sequences[0])
