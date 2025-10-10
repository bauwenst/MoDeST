"""
For downloading files from the MorphyNet repository on GitHub.

URLs for MorphyNet have the following format:
    https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{lang}/{lang}.{inflectional/derivational}.v1.tsv
"""
from enum import Enum
from pathlib import Path
from typing import Iterable, Any, Iterator
from abc import abstractmethod

import langcodes
import requests
from logging import getLogger

from ..formats.morphynet import MorphyNetInflection, MorphyNetDerivation
from ..formats.tsv import iterateTsv
from ..interfaces.datasets import ModestDataset, M
from ..interfaces.kernels import ModestKernel, Raw

logger = getLogger(__name__)

MORPHYNET_LANGUAGES = {
    langcodes.find("Catalan"): "cat",
    langcodes.find("Czech"): "ces",
    langcodes.find("German"): "deu",
    langcodes.find("English"): "eng",
    langcodes.find("Finnish"): "fin",
    langcodes.find("French"): "fra",

    langcodes.find("Serbo-Croatian"): "hbs",
    langcodes.find("Bosnian")       : "hbs",
    langcodes.find("Croatian")      : "hbs",
    langcodes.find("Montenegrin")   : "hbs",
    langcodes.find("Serbian")       : "hbs",

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

    @abstractmethod
    def getSubset(self) -> MorphynetSubset:
        pass

    def getCollectionName(self) -> str:
        return "MorphyNet"

    def _files(self) -> list[Path]:
        morphynet_code = MORPHYNET_LANGUAGES.get(self.getLanguage())
        if morphynet_code is None:
            raise ValueError(f"Language not in MorphyNet: {self.getLanguage()}")

        cache_path = self._getCachePath() / f"{morphynet_code}.{self.getSubset().toString()}.v1.tsv"
        if not cache_path.exists():
            url = f"https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{morphynet_code}/{self._getRemoteFilename(morphynet_code, self.getSubset())}"
            response = requests.get(url)
            if response.status_code == 404:
                raise RuntimeError(f"The URL {url} does not exist.")
            with open(cache_path, "wb") as handle:
                handle.write(response.content)

        return [cache_path]

    def _getRemoteFilename(self, langcode: str, subset: MorphynetSubset) -> str:
        return f"{langcode}.{subset.toString()}.v1.tsv"


class MorphyNetDataset_Inflection(MorphyNetDataset[MorphyNetInflection]):

    def __init__(self, verbose: bool=False, skip_if_unknown: bool=False):
        super().__init__()
        self._verbose = verbose
        self._skip_if_unknown = skip_if_unknown

    def getSubset(self) -> MorphynetSubset:
        return MorphynetSubset.INFLECTIONAL

    def _kernels(self) -> list[ModestKernel[Any,MorphyNetInflection]]:
        return [_MorphyNetKernel_Inflection(verbose=self._verbose, skip_if_unknown=self._skip_if_unknown)]


class _MorphyNetKernel_Inflection(ModestKernel[tuple[str,str,str,str],MorphyNetInflection]):

    def __init__(self, verbose: bool, skip_if_unknown: bool):
        self._verbose = verbose
        self._skip_if_unknown = skip_if_unknown

    def _generateRaw(self, path: Path):
        yield from enumerate(iterateTsv(path, verbose=self._verbose))

    def _parseRaw(self, raw, id: int):
        raise NotImplementedError()  # We don't implement example-per-example parsing because they depend on each other.

    def _generateObjects(self, path: Path):
        prev = None
        seen = set()

        for id, parts in self._generateRaw(path):
            try:
                lemma, word, tag, decomposition = parts
                word = word.split(" ")[-1]
                if "," in word or "''" in word or "(" in word:  # This happens e.g. in a couple Czech entries, although most are fixed after you split on spaces.
                    raise
            except:
                line = '\t'.join(parts).strip()
                logger.info(f"Bad MorphyNet line: {line if line else '(empty)'}")
                continue

            # There are duplicate decompositions in MorphyNet (either with a different tag, or literally just duplicate entries in the dataset).
            compressed = (hash(decomposition), lemma[0:3])  # lemma reduces the possibility of collisions to basically 0.
            if compressed in seen:
                continue
            seen.add(compressed)

            curr = MorphyNetInflection(
                id=id,
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
                if not self._skip_if_unknown:
                    try:
                        logger.info(f"Last word '{prev.word}' had no morphemes. Will be imputed by looking forwards at {curr.word} -> {curr.morphemes}")
                        self._imputeMorphemes(with_known_morphemes=curr, with_unknown_morphemes=prev)
                        logger.info(f"\t{prev.morphemes}")
                    except:
                        logger.info(f"\tFAILED; {prev.word} removed from dataset.")
                        prev = curr
                        continue
                else:
                    prev = curr
                    continue
            elif curr.decompose() == tuple("-"):  # The fact that this is in an 'else' implies that prev has morphemes, which means even when this branch does nothing, you can output it.
                if not self._skip_if_unknown:
                    try:
                        logger.info(f"Current word '{curr.word}' has no morphemes, but the previous word did. Will be imputed by looking backwards at {prev.word} -> {prev.morphemes}")
                        self._imputeMorphemes(with_known_morphemes=prev, with_unknown_morphemes=curr)
                        logger.info(f"\t{curr.morphemes}")
                    except:
                        logger.info(f"\tFAILED; trying again soon.")
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
            logger.info("\tWARNING: removed only part of the found stem. May have removed too much stem.")

        suffix = with_unknown_morphemes.word.removeprefix(stem)
        if suffix:
            with_unknown_morphemes.morphemes = known_morphemes[:-1] + (suffix,)
        else:
            with_unknown_morphemes.morphemes = known_morphemes[:-1]


class MorphyNetDataset_Derivation(MorphyNetDataset[MorphyNetDerivation]):

    def __init__(self, verbose: bool=False):
        super().__init__()
        self._verbose = verbose

    def getSubset(self) -> MorphynetSubset:
        return MorphynetSubset.DERIVATIONAL

    def _kernels(self) -> list[ModestKernel[Any,MorphyNetDerivation]]:
        return [_MorphyNetKernel_Derivation(verbose=self._verbose)]


class _MorphyNetKernel_Derivation(ModestKernel[tuple[str,str,str,str,str,str],MorphyNetDerivation]):

    def __init__(self, verbose: bool=False):
        self._verbose = verbose

    def _generateRaw(self, path: Path):
        yield from enumerate(iterateTsv(path, verbose=self._verbose))

    def _parseRaw(self, raw, id: int):
        original, result, original_pos, result_pos, affix, affix_type = raw
        return MorphyNetDerivation(
            id=id,
            word=result,
            base=original,
            affix=affix,
            prefix_not_suffix=(affix_type == "prefix")
        )

    def _createWriter(self, path: Path):
        raise NotImplementedError()
