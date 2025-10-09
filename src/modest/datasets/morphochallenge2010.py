from typing import Iterable
from pathlib import Path

import langcodes
import requests
from logging import getLogger

from ..interfaces.datasets import ModestDataset, Languageish
from ..formats.tsv import iterateHandle
from ..formats.morphochallenge2010 import MorphoChallenge2010Morphology

LOGGER = getLogger(__name__)

MC_LANGUAGES = {
    langcodes.find("English"): "eng",
    langcodes.find("Finnish"): "fin",
    # langcodes.find("German"): "ger",  # TODO: Only has decompositions, no segmentations. That means you need a simpler parser but have the same information.
    langcodes.find("Turkish"): "tur"
}


class MorphoChallenge2010Dataset(ModestDataset[MorphoChallenge2010Morphology]):

    def __init__(self, verbose: bool=False):
        super().__init__()
        self._verbose = verbose

    def getName(self) -> str:
        return "MC2010"

    def _get(self) -> Path:
        code = MC_LANGUAGES.get(self.getLanguage())
        if code is None:
            raise ValueError(f"Unknown language: {self.getLanguage()}")

        cache = self._getCachePath() / f"{code}.segmentation.train.tsv"
        if not cache.exists():
            url = f"http://morpho.aalto.fi/events/morphochallenge2010/data/goldstd_trainset.segmentation.{code}"
            response = requests.get(url)
            with open(cache, "wb") as handle:
                handle.write(response.content)

        return cache

    def _generate(self, path: Path) -> Iterable[MorphoChallenge2010Morphology]:
        is_turkish = (self.getLanguage() == langcodes.find("Turkish"))

        with open(path, "r", encoding="windows-1252") as handle:
            for line in iterateHandle(handle, verbose=self._verbose):
                lemma, tag = line.split("\t")
                try:
                    yield MorphoChallenge2010Morphology(
                        word=lemma,
                        segmentation_tag=tag,
                        turkish_to_utf8=is_turkish
                    )
                except GeneratorExit:
                    raise GeneratorExit
                except:
                    LOGGER.warning(f"Failed to parse morphology: '{lemma}' tagged as '{tag}'")
