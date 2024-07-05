from pathlib import Path
import langcodes
import requests

from .paths import PathManagement


MC_LANGUAGES = {
    langcodes.find("English"): "eng",
    langcodes.find("Finnish"): "fin",
    # langcodes.find("German"): "ger",  # TODO: Only has decompositions, no segmentations. That means you need a simpler parser but have the same information.
    langcodes.find("Turkish"): "tur"
}


class MorphoChallenge2010Downloader:

    def get(self, language: langcodes.Language) -> Path:
        code = MC_LANGUAGES.get(language)
        if code is None:
            raise ValueError("Unknown language:", language)

        cache = PathManagement.datasetCache(language=language, dataset_name="MC2010") / f"{code}.segmentation.train.tsv"
        if cache.exists():
            url = f"http://morpho.aalto.fi/events/morphochallenge2010/data/goldstd_trainset.segmentation.{code}"
            response = requests.get(url)
            with open(cache, "wb") as handle:
                handle.write(response.content)

        return cache
