from tktkt.preparation.mappers import MorphoChallengeCapitals
from ..interfaces.morphologies import WordDecomposition


class MorphoChallenge2010Morphology(WordDecomposition):

    def __init__(self, id: int, word: str, segmentation_tag: str, turkish_to_utf8: bool=True):
        if turkish_to_utf8:
            mapping = MorphoChallengeCapitals()  # Only for Turkish, but since it works with capitals and MC2010 lowercases everything else, this isn't a problem.
            word             = mapping.invert(word)
            segmentation_tag = mapping.invert(segmentation_tag)

        super().__init__(id=id, word=word)

        self.morph_sequences    = []
        self.morpheme_sequences = []
        self.tag_sequences      = []

        for structure in segmentation_tag.split(","):
            morphs = []
            morphemes = []
            tags = []
            for mm in structure.strip().split(" "):
                morph, morpheme = mm.split(":")

                # Morph
                if morph != "~":
                    morphs.append(morph)

                # Morpheme
                if morpheme == "~":
                    morpheme, tag = "-", ""
                elif "_" in morpheme:
                    underscore_separated = morpheme.split("_")
                    morpheme, tag = "_".join(underscore_separated[:-1]), underscore_separated[-1]
                else:
                    morpheme, tag = "", morpheme

                if morpheme != "":  # Not sure if it makes sense to leave out empty morphemes, since the tags will no longer be alignable.
                    morphemes.append(morpheme)
                tags.append(tag)

            self.morph_sequences.append(morphs)
            self.morpheme_sequences.append(morphemes)
            self.tag_sequences.append(tags)

    def segment(self) -> tuple[str, ...]:
        return tuple(self.morph_sequences[0])

    def decompose(self) -> tuple[str, ...]:
        return tuple(self.morpheme_sequences[0])
