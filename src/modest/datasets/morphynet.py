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
        prev: MorphyNetInflection = None
        for parts in iterateTsv(path):
            lemma, word, tag, decomposition = parts
            curr = MorphyNetInflection(
                word=word,
                raw_morpheme_sequence=decomposition,
                lemma=lemma,
                lexical_tag=tag
            )
            if curr.word.startswith("no "):  # Skip this while keeping the previous object ready.
                continue

            # If this is the first good word, put it in the chamber for now.
            if prev is None:
                prev = curr
                continue

            # Impute the previous decomp if it doesn't exist.
            if prev.decompose() == tuple("-"):
                try:
                    print(f"Last word '{prev.word}' had no morphemes. Will be imputed by looking forwards at", curr.word, "->", curr.morphemes)
                    self._imputeMorphemes(with_known_morphemes=curr, with_unknown_morphemes=prev)
                    print("\t", prev.morphemes)
                except:
                    print(f"\tFAILED; {prev.word} removed from dataset.")
                    prev = curr
                    continue
            elif curr.decompose() == tuple("-"):  # The fact that this is in an 'else' implies that prev has morphemes, which means even when this branch does nothing, you can output it.
                try:
                    print(f"Current word '{curr.word}' has no morphemes, but the previous word did. Will be imputed by looking backwards at", prev.word, "->", prev.morphemes)
                    self._imputeMorphemes(with_known_morphemes=prev, with_unknown_morphemes=curr)
                    print("\t", curr.morphemes)
                except:
                    print(f"\tFAILED; trying again soon.")
                    # Still output prev.

            # Output
            yield prev
            prev = curr

        if prev is not None:
            yield prev

    def _imputeMorphemes(self, with_known_morphemes: MorphyNetInflection, with_unknown_morphemes: MorphyNetInflection):
        known_morphemes = with_known_morphemes.decompose()
        assert known_morphemes != tuple("-")
        assert len(known_morphemes) == 2  # TODO: Needs to be loosened to allow e.g. inferring  'desleídme' -> ('desleid', 'me')  (where 'desleid' itself has a previous decomposition known)   from   desliámoslas -> ('desleír', 'amos', 'las'), where you find the stem by looking at only the first morph because all the inflection is done through suffixation.
        assert with_known_morphemes.lemma == with_unknown_morphemes.lemma

        # 1. Let Viterbi find the morph corresponding to the last morpheme of the next word.
        last_morph_in_curr = with_known_morphemes.segment()[-1]

        # 2. Trim that from that word to reveal a stem.
        stem = with_known_morphemes.word.removesuffix(last_morph_in_curr)

        # 3. The first word's suffix is whatever is after that stem.
        uncertain = False
        while stem and not with_unknown_morphemes.word.startswith(stem):
            stem = stem[:-1]
            uncertain = True
        assert stem
        if uncertain:
            print("\tWARNING: removed only part of the found stem. May have removed too much stem.")

        suffix = with_unknown_morphemes.word.removeprefix(stem)
        if suffix:
            with_unknown_morphemes.morphemes = known_morphemes[:-1] + (suffix,)
        else:
            with_unknown_morphemes.morphemes = known_morphemes[:-1]


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
