from typing import Iterable, Iterator, Any
from pathlib import Path

import requests
from logging import getLogger

from ..interfaces.readers import ModestReader, Raw, M

logger = getLogger(__name__)

from tktkt.util.types import L

from ..interfaces.datasets import ModestDataset, Languageish
from ..formats.tsv import iterateHandle
from ..formats.morphochallenge2010 import MorphoChallenge2010Morphology


MC_LANGUAGES = {
    L("English"): "eng",
    L("Finnish"): "fin",
    # L("German"): "ger",  # TODO: Only has decompositions, no segmentations. That means you need a simpler parser but have the same information.
    L("Turkish"): "tur"
}


class MorphoChallenge2010Dataset(ModestDataset[MorphoChallenge2010Morphology]):

    def __init__(self, verbose: bool=False):
        super().__init__()
        self._verbose = verbose

    def getCollectionName(self) -> str:
        return "MC2010"

    def _readers(self) -> list[ModestReader[Any,MorphoChallenge2010Morphology]]:
        return [_MorphoChallengeReader(verbose=self._verbose, is_turkish=self.getLanguage() == L("Turkish"))]

    def _files(self) -> list[Path]:
        code = MC_LANGUAGES.get(self.getLanguage())
        if code is None:
            raise ValueError(f"Unknown language: {self.getLanguage()}")

        cache = self._getCachePath() / f"{code}.segmentation.train.tsv"
        if not cache.exists():
            url = f"http://morpho.aalto.fi/events/morphochallenge2010/data/goldstd_trainset.segmentation.{code}"
            response = requests.get(url)
            with open(cache, "wb") as handle:
                handle.write(response.content)

        return [cache]


class _MorphoChallengeReader(ModestReader[str, MorphoChallenge2010Morphology]):

    def __init__(self, verbose: bool, is_turkish: bool):
        self._verbose = verbose
        self._is_turkish = is_turkish

    def _generateRaw(self, path: Path) -> Iterator[tuple[int,str]]:
        with open(path, "r", encoding="windows-1252") as handle:
            yield from iterateHandle(handle, verbose=self._verbose)

    def _parseRaw(self, raw: str, id: int) -> MorphoChallenge2010Morphology:
        lemma, tag = raw.split("\t")
        try:
            return MorphoChallenge2010Morphology(
                id=id,
                word=lemma,
                segmentation_tag=tag,
                turkish_to_utf8=self._is_turkish
            )
        except:
            logger.info(f"Failed to parse morphology: '{lemma}' tagged as '{tag}'")
            raise RuntimeError()

    def _createWriter(self):
        raise NotImplementedError()
