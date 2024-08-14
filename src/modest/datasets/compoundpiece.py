"""
For downloading files from the HuggingFace hub.
"""
import datasets
import langcodes
from pathlib import Path
from typing import Iterable

from ..formats.trivial import TrivialDecomposition
from ..interfaces.datasets import ModestDataset
from ..paths import PathManagement
from ..formats.tsv import iterateTsv


class CompoundPieceDataset(ModestDataset[TrivialDecomposition]):

    LANGCODES = {
        langcodes.find("Afrikaans"): "af",
        langcodes.find("Azerbaijani"): "az",
        langcodes.find("Belarusian"): "be",
        langcodes.find("Bulgarian"): "bg",
        langcodes.find("Bengali"): "bn",
        langcodes.find("Catalan"): "ca",
        langcodes.find("Czech"): "cs",
        langcodes.find("Welsh"): "cy",
        langcodes.find("Danish"): "da",
        langcodes.find("German"): "de",
        langcodes.find("Greek"): "el",
        langcodes.find("English"): "en",
        langcodes.find("Esperanto"): "eo",
        langcodes.find("Spanish"): "es",
        langcodes.find("Estonian"): "et",
        langcodes.find("Basque"): "eu",
        langcodes.find("Persian"): "fa",
        langcodes.find("Finnish"): "fi",
        langcodes.find("French"): "fr",
        langcodes.find("Western-Frisian"): "fy",
        langcodes.find("Irish"): "ga",
        langcodes.find("Galician"): "gl",
        langcodes.find("Gujarati"): "gu",
        langcodes.find("Hebrew"): "he",
        langcodes.find("Hindi"): "hi",
        langcodes.find("Hungarian"): "hu",
        langcodes.find("Armenian"): "hy",
        langcodes.find("Indonesian"): "id",
        langcodes.find("Icelandic"): "is",
        langcodes.find("Italian"): "it",
        langcodes.find("Georgian"): "ka",
        langcodes.find("Kazakh"): "kk",
        langcodes.find("Kirghiz"): "ky",
        langcodes.find("Latin"): "la",
        langcodes.find("Lithuanian"): "lt",
        langcodes.find("Latvian"): "lv",
        langcodes.find("Malagasy"): "mg",
        langcodes.find("Macedonian"): "mk",
        langcodes.find("Malayalam"): "ml",
        langcodes.find("Maltese"): "mt",
        langcodes.find("Dutch"): "nl",
        langcodes.find("Panjabi"): "pa",
        langcodes.find("Polish"): "pl",
        langcodes.find("Portuguese"): "pt",
        langcodes.find("Romanian"): "ro",
        langcodes.find("Russian"): "ru",
        langcodes.find("Slovak"): "sk",
        langcodes.find("Albanian"): "sq",
        langcodes.find("Swedish"): "sv",
        langcodes.find("Tamil"): "ta",
        langcodes.find("Telugu"): "te",
        langcodes.find("Thai"): "th",
        langcodes.find("Turkish"): "tr",
        langcodes.find("Ukrainian"): "uk",
        langcodes.find("Yiddish"): "yi",
        langcodes.find("Yoruba"): "yo"
    }

    def __init__(self, language: langcodes.Language):
        super().__init__(name="CompoundPiece", language=language)

    def _get(self) -> Path:
        langcode = CompoundPieceDataset.LANGCODES.get(self._language)
        if langcode is None:
            raise ValueError(f"Unknown language: {self._language}")

        cache = self._getCachePath() / f"{langcode}.S1-S2.tsv"
        if not cache.exists():
            data = datasets.load_dataset("benjamin/compoundpiece", "wiktionary")["train"]
            with open(cache, "w", encoding="utf-8") as handle:
                for row in data.filter(lambda row: row["lang"] == langcode):
                    handle.write(row["word"] + "\t" + row["norm"] + "\t" + row["segmentation"] + "\n")

        return cache

    def _generate(self, path: Path, **kwargs) -> Iterable[TrivialDecomposition]:
        for word, decomposition, segmentation in iterateTsv(path):
            yield TrivialDecomposition(
                word=word,
                decomposition_tag=decomposition,
                segmentation_tag=segmentation,
                sep="-"
            )