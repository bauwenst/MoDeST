"""
For downloading files from the HuggingFace hub.
"""
from typing import Any

import datasets
from pathlib import Path

from tktkt.util.types import L

from ..formats.trivial import TrivialDecomposition
from ..interfaces.datasets import ModestDataset, Languageish, M
from ..formats.tsv import iterateTsv
from ..interfaces.kernels import ModestKernel


class CompoundPieceDataset(ModestDataset[TrivialDecomposition]):

    LANGCODES = {
        L("Afrikaans"): "af",
        L("Azerbaijani"): "az",
        L("Belarusian"): "be",
        L("Bulgarian"): "bg",
        L("Bengali"): "bn",
        L("Catalan"): "ca",
        L("Czech"): "cs",
        L("Welsh"): "cy",
        L("Danish"): "da",
        L("German"): "de",
        L("Greek"): "el",
        L("English"): "en",
        L("Esperanto"): "eo",
        L("Spanish"): "es",
        L("Estonian"): "et",
        L("Basque"): "eu",
        L("Persian"): "fa",
        L("Finnish"): "fi",
        L("French"): "fr",
        L("Western-Frisian"): "fy",
        L("Irish"): "ga",
        L("Galician"): "gl",
        L("Gujarati"): "gu",
        L("Hebrew"): "he",
        L("Hindi"): "hi",
        L("Hungarian"): "hu",
        L("Armenian"): "hy",
        L("Indonesian"): "id",
        L("Icelandic"): "is",
        L("Italian"): "it",
        L("Georgian"): "ka",
        L("Kazakh"): "kk",
        L("Kirghiz"): "ky",
        L("Latin"): "la",
        L("Lithuanian"): "lt",
        L("Latvian"): "lv",
        L("Malagasy"): "mg",
        L("Macedonian"): "mk",
        L("Malayalam"): "ml",
        L("Maltese"): "mt",
        L("Dutch"): "nl",
        L("Panjabi"): "pa",
        L("Polish"): "pl",
        L("Portuguese"): "pt",
        L("Romanian"): "ro",
        L("Russian"): "ru",
        L("Slovak"): "sk",
        L("Albanian"): "sq",
        L("Swedish"): "sv",
        L("Tamil"): "ta",
        L("Telugu"): "te",
        L("Thai"): "th",
        L("Turkish"): "tr",
        L("Ukrainian"): "uk",
        L("Yiddish"): "yi",
        L("Yoruba"): "yo"
    }

    def __init__(self, language: Languageish):  # If you ever add these as separate datasets to be imported, this constructor probably needs to be removed.
        super().__init__()
        self._lang = language

    def getCollectionName(self) -> str:
        return "CompoundPiece"

    def _getLanguage(self) -> Languageish:
        return self._lang

    def _kernels(self) -> list[ModestKernel[Any,M]]:
        return [_CompoundPieceKernel()]

    def _files(self) -> list[Path]:
        langcode = CompoundPieceDataset.LANGCODES.get(self.getLanguage())
        if langcode is None:
            raise ValueError(f"Unknown language: {self.getLanguage()}")

        cache = self._getCachePath() / f"{langcode}.S1-S2.tsv"
        if not cache.exists():
            data = datasets.load_dataset("benjamin/compoundpiece", "wiktionary")["train"]
            with open(cache, "w", encoding="utf-8") as handle:
                for row in data.filter(lambda row: row["lang"] == langcode):
                    handle.write(row["word"] + "\t" + row["norm"] + "\t" + row["segmentation"] + "\n")

        return [cache]


class _CompoundPieceKernel(ModestKernel[tuple[str,str,str],TrivialDecomposition]):

    def _generateRaw(self, path: Path):
        yield from iterateTsv(path)

    def _parseRaw(self, raw, id: int):
        word, decomposition, segmentation = raw
        yield TrivialDecomposition(
            id=id,
            word=word,
            decomposition_tag=decomposition,
            segmentation_tag=segmentation,
            sep="-"
        )

    def _createWriter(self):
        raise NotImplementedError()
