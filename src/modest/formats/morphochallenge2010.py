from typing import Iterable
from pathlib import Path

from tktkt.preparation.mappers import MorphoChallengeCapitals
from ..interfaces.formative_lemmata import LemmaMorphology
from ..formats.tsv_wordfrequency import iterateHandle


class MorphoChallenge2010LemmaMorphology(LemmaMorphology):

    def __init__(self, lemma: str, segmentation_tag: str, turkish_to_utf8: bool=True):
        if turkish_to_utf8:
            mapping = MorphoChallengeCapitals()  # Only for Turkish, but since it works with capitals and MC2010 lowercases everything else, this isn't a problem.
            lemma            = mapping.invert(lemma)
            segmentation_tag = mapping.invert(segmentation_tag)

        self._lemma = lemma
        self.morph_sequences    = []
        self.morpheme_sequences = []

        for structure in segmentation_tag.split(","):
            morphs = []
            morphemes = []
            for mm in structure.strip().split(" "):
                morph, morpheme = mm.split(":")
                morphs.append(morph)
                morphemes.append(morpheme)

            self.morph_sequences.append(morphs)
            self.morpheme_sequences.append(morphemes)

    def lemma(self) -> str:
        return self._lemma

    def morphSplit(self) -> str:
        return " ".join(self.morph_sequences[0])

    def morphemeSplit(self) -> str:
        return " ".join(self.morpheme_sequences[0])

    def lexemeSplit(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def generator(file: Path, verbose=False) -> Iterable["LemmaMorphology"]:
        with open(file, "r", encoding="windows-1252") as handle:
            for line in iterateHandle(handle, verbose=verbose):
                lemma, tag = line.split("\t")
                try:
                    yield MorphoChallenge2010LemmaMorphology(lemma=lemma, segmentation_tag=tag)
                except:
                    print(f"Failed to parse morphology: '{lemma}' tagged as '{tag}'")
