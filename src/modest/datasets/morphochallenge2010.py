from pathlib import Path
import langcodes
import requests
from typing import Iterable, Generic

from modest.paths import PathManagement
from ..interfaces.datasets import ModestDataset
from ..formats.tsv import iterateHandle
from ..formats.morphochallenge2010 import MorphoChallenge2010Morphology


MC_LANGUAGES = {
    langcodes.find("English"): "eng",
    langcodes.find("Finnish"): "fin",
    # langcodes.find("German"): "ger",  # TODO: Only has decompositions, no segmentations. That means you need a simpler parser but have the same information.
    langcodes.find("Turkish"): "tur"
}


class MorphoChallenge2010Dataset(ModestDataset):

    def __init__(self, language: langcodes.Language):
        self.language = language

    def _get(self) -> Path:
        code = MC_LANGUAGES.get(self.language)
        if code is None:
            raise ValueError(f"Unknown language: {self.language}")

        cache = PathManagement.datasetCache(language=self.language, dataset_name="MC2010") / f"{code}.segmentation.train.tsv"
        if not cache.exists():
            url = f"http://morpho.aalto.fi/events/morphochallenge2010/data/goldstd_trainset.segmentation.{code}"
            response = requests.get(url)
            with open(cache, "wb") as handle:
                handle.write(response.content)

        return cache

    def _generate(self, path: Path) -> Iterable[MorphoChallenge2010Morphology]:
        is_turkish = (self.language == langcodes.find("Turkish"))

        with open(path, "r", encoding="windows-1252") as handle:
            for line in iterateHandle(handle):
                lemma, tag = line.split("\t")
                try:
                    yield MorphoChallenge2010Morphology(
                        word=lemma,
                        segmentation_tag=tag,
                        turkish_to_utf8=is_turkish
                    )
                except:
                    print(f"Failed to parse morphology: '{lemma}' tagged as '{tag}'")