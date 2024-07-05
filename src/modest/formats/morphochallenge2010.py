from typing import Tuple

from tktkt.preparation.mappers import MorphoChallengeCapitals
from ..interfaces.morphologies import WordDecomposition


class MorphoChallenge2010LemmaMorphology(WordDecomposition):

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


####################################################################################################


from typing import Iterable
from pathlib import Path
import langcodes

from .tsv import iterateHandle
from ..interfaces.datasets import ModestDataset
from ..downloaders.morphochallenge2010 import MorphoChallenge2010Downloader


class MorphoChallenge2010Dataset(ModestDataset):

    def __init__(self, language: langcodes.Language):
        self.language = language

    def _load(self) -> Path:
        dl = MorphoChallenge2010Downloader()
        return dl.get(language=self.language)

    def _generate(self, path: Path) -> Iterable[MorphoChallenge2010LemmaMorphology]:
        is_turkish = (self.language == langcodes.find("Turkish"))

        with open(path, "r", encoding="windows-1252") as handle:
            for line in iterateHandle(handle):
                lemma, tag = line.split("\t")
                try:
                    yield MorphoChallenge2010LemmaMorphology(
                        word=lemma,
                        segmentation_tag=tag,
                        turkish_to_utf8=is_turkish
                    )
                except:
                    print(f"Failed to parse morphology: '{lemma}' tagged as '{tag}'")
