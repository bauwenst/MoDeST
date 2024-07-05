"""
For downloading files from the MorphyNet repository on GitHub.

URLs for MorphyNet have the following format:
    https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{lang}/{lang}.{inflectional/derivational}.v1.tsv
"""
from enum import Enum
from pathlib import Path

import langcodes
import requests

from .paths import PathManagement

MORPHYNET_LANGUAGES = {
    langcodes.find("Catalan"): "cat",
    langcodes.find("Czech"): "ces",
    langcodes.find("German"): "deu",
    langcodes.find("English"): "eng",
    langcodes.find("Finnish"): "fin",
    langcodes.find("French"): "fra",
    langcodes.find("Serbo-Croatian"): "hbs",  # Possibly you want langcodes.find("Serbian") and langcodes.find("Croatian") to resolve to this code too.
    langcodes.find("Hungarian"): "hun",
    langcodes.find("Italian"): "ita",
    langcodes.find("Mongolian"): "mon",
    langcodes.find("Polish"): "pol",
    langcodes.find("Portuguese"): "por",
    langcodes.find("Russian"): "rus",
    langcodes.find("Spanish"): "spa",
    langcodes.find("Swedish"): "swe"
}


class MorphynetSubset(Enum):
    INFLECTIONAL = 1
    DERIVATIONAL = 2

    def toString(self) -> str:
        if self == MorphynetSubset.INFLECTIONAL:
            return "inflectional"
        elif self == MorphynetSubset.DERIVATIONAL:
            return "derivational"
        else:
            raise ValueError("Enum value has no string representation:", self)


class MorphynetDownloader:

    def get(self, language: langcodes.Language, subset: MorphynetSubset) -> Path:
        morphynet_code = MORPHYNET_LANGUAGES.get(language)
        if morphynet_code is None:
            raise ValueError(f"Language not in MorphyNet: {language}")

        cache_path = PathManagement.datasetCache(language, "MorphyNet") / f"{morphynet_code}.{subset.toString()}.v1.tsv"
        if not cache_path.exists():
            url = f"https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{morphynet_code}/{cache_path.name}"
            response = requests.get(url)
            with open(cache_path, "wb") as handle:
                handle.write(response.content)

        return cache_path
