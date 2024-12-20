"""
For downloading files from the MorphyNet repository on GitHub.

URLs for MorphyNet have the following format:
    https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{lang}/{lang}.{inflectional/derivational}.v1.tsv
"""
from enum import Enum
from pathlib import Path
from typing import Iterable

import langcodes
import requests

from ..formats.morphynet import MorphyNetInflection, MorphyNetDerivation
from ..formats.tsv import iterateTsv
from ..interfaces.datasets import ModestDataset, M, Languageish


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


class MorphyNetDataset(ModestDataset[M]):
    """
    Has two subsets (inflectional and derivational) which can share their download code but
    need different generator code.
    """

    def __init__(self, language: Languageish, subset: MorphynetSubset):
        super().__init__(name="MorphyNet", language=language)
        self._subset = subset

    def _get(self) -> Path:
        morphynet_code = MORPHYNET_LANGUAGES.get(self._language)
        if morphynet_code is None:
            raise ValueError(f"Language not in MorphyNet: {self._language}")

        cache_path = self._getCachePath() / f"{morphynet_code}.{self._subset.toString()}.v1.tsv"
        if not cache_path.exists():
            url = f"https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{morphynet_code}/{self._getRemoteFilename(morphynet_code, self._subset)}"
            response = requests.get(url)
            with open(cache_path, "wb") as handle:
                handle.write(response.content)

        return cache_path

    def _getRemoteFilename(self, langcode: str, subset: MorphynetSubset) -> str:
        return f"{langcode}.{subset.toString()}.v1.tsv"


class MorphyNetDataset_Inflection(MorphyNetDataset[MorphyNetInflection]):

    def __init__(self, language: Languageish):
        super().__init__(language=language, subset=MorphynetSubset.INFLECTIONAL)

    def _generate(self, path: Path, **kwargs) -> Iterable[MorphyNetInflection]:
        for parts in iterateTsv(path):
            lemma, word, tag, decomposition = parts
            yield MorphyNetInflection(
                word=word,
                raw_morpheme_sequence=decomposition,
                lemma=lemma,
                lexical_tag=tag
            )


class MorphyNetDataset_Derivation(MorphyNetDataset[MorphyNetDerivation]):

    def __init__(self, language: Languageish):
        super().__init__(language=language, subset=MorphynetSubset.DERIVATIONAL)

    def _generate(self, path: Path, **kwargs) -> Iterable[MorphyNetDerivation]:
        for parts in iterateTsv(path):
            original, result, original_pos, result_pos, affix, affix_type = parts
            try:
                yield MorphyNetDerivation(
                    word=result,
                    base=original,
                    affix=affix,
                    prefix_not_suffix=(affix_type == "prefix")
                )
            except:
                print("Unparsable MorphyNet derivation:", parts)
                pass
