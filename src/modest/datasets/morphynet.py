"""
For downloading files from the MorphyNet repository on GitHub.

URLs for MorphyNet have the following format:
    https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{lang}/{lang}.{inflectional/derivational}.v1.tsv
"""
from typing import Iterable, Any, Iterator
from abc import abstractmethod
from pathlib import Path
from enum import Enum

import langcodes
import requests
from logging import getLogger

from ..formats.morphynet import MorphyNetInflection, MorphyNetDerivation
from ..formats.tsv import iterateTsv
from ..interfaces.datasets import ModestDataset, M
from ..interfaces.readers import ModestReader, Raw, Writer
from ..transformations.precompute import TsvWriter

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
    """
    For MorphyNet inflections. Note that these do not necessarily give a full morphological decomposition: they only give
    an inflectional decomposition. For example, in the German dataset, "Weltwirtschaft" ("global economy") is said to have
    no decomposition despite clearly consisting of "Welt/wirt/schaft".
    """

    def __init__(self, verbose: bool=False, skip_if_unknown: bool=False, skip_if_unimputable: bool=False):
        """
        :param skip_if_unknown: Whether to discard the entries that have no decomposition given by MorphyNet, which includes
                                e.g. masculine singular adjectives, infinitives, uninflected nouns, ...
                                If True, you will lose a lot of
        :param skip_if_unimputable: Whether to discard the entries that have no decomposition but only if they can't be
                                    imputed. If True, you will lose e.g. adjectives whose masculine and feminine are equal,
                                    nouns that are not inflected, and so in general, monomorphemic words (at least inflectionally).
        """
        super().__init__()
        self._verbose = verbose
        self._skip_if_unknown = skip_if_unknown
        self._skip_if_unimputable = skip_if_unimputable

    def getSubset(self) -> MorphynetSubset:
        return MorphynetSubset.INFLECTIONAL

    def _readers(self) -> list[ModestReader[Any,MorphyNetInflection]]:
        return [_MorphyNetReader_Inflection(verbose=self._verbose, skip_if_unknown=self._skip_if_unknown, skip_if_unimputable=self._skip_if_unimputable)]


class _MorphyNetReader_Inflection(ModestReader[tuple[str,str,str,str],MorphyNetInflection]):

    def __init__(self, verbose: bool, skip_if_unknown: bool, skip_if_unimputable: bool):
        self._verbose = verbose
        self._skip_if_unknown = skip_if_unknown
        self._skip_if_unimputable = skip_if_unimputable

    def _generateRaw(self, path: Path):
        yield from iterateTsv(path, verbose=self._verbose)

    def _createWriter(self):
        return TsvWriter()

    def _parseRaw(self, raw, id: int):
        raise NotImplementedError()  # We don't implement example-per-example parsing because they depend on each other.

    def generateObjects(self, path: Path):
        prev = None
        seen = set()

        for id, parts in self.generateRaw(path):
            try:
                lemma, word, tag, decomposition = parts
                word = word.split(" ")[-1]
                if "," in word or "''" in word or "(" in word:  # This happens e.g. in a couple Czech entries, although most are fixed after you split on spaces.
                    raise
            except:
                line = '\t'.join(parts).strip()
                logger.info(f"Bad MorphyNet line: {line if line else '(empty)'}")
                continue

            curr = MorphyNetInflection(
                id=id,
                word=word,
                raw_morpheme_sequence=decomposition,
                lemma=lemma,
                lexical_tag=tag
            )
            # if word.startswith("no "):  # Skip this while keeping the previous object ready.
            #     continue

            # If this is the first entry, put it in the chamber for now.
            if prev is None:
                prev = curr
                continue

            # Impute the previous decomp if it doesn't exist.
            discard_prev = False
            if prev.decompose() == ("-",):
                if self._skip_if_unknown:
                    discard_prev = True
                else:
                    try:
                        logger.info(f"Last word '{prev.word}' had no morphemes. Will be imputed by looking forwards at {curr.word} -> {curr.morphemes}")
                        self._imputeMorphemes(with_known_morphemes=curr, with_unknown_morphemes=prev)
                        logger.info(f"\tFound: {prev.morphemes}")
                    except:
                        if self._skip_if_unimputable:
                            discard_prev = True
                            logger.info(f"\tFailed. {prev.word} removed from dataset.")
                        else:  # We embrace the fact that there is no decomposition and assume that
                            prev.morphemes = (prev.word,)
                            logger.info(f"\tFailed. Let's assume it is just one big morpheme.")
            elif curr.decompose() == ("-",):  # The fact that this is in an 'else' implies that prev has morphemes, which means even when this branch does nothing, you can output it.
                if self._skip_if_unknown:
                    pass  # Don't try to impute, and you'll default to the first block above in the next iteration.
                else:
                    try:
                        logger.info(f"Current word '{curr.word}' has no morphemes, but the previous word did. Will be imputed by looking backwards at {prev.word} -> {prev.morphemes}")
                        self._imputeMorphemes(with_known_morphemes=prev, with_unknown_morphemes=curr)
                        logger.info(f"\t{curr.morphemes}")
                    except:
                        logger.info(f"\tFAILED; trying again soon.")

            if not discard_prev:
                # There are duplicate decompositions in MorphyNet (either with a different tag, or literally just duplicate entries in the dataset).
                fingerprint = (hash(decomposition), hash(lemma))  # In case DIFFERENT decompositions (e.g. a|b|c and e|f|g) have the same hash (highly, highly unlikely), we use the lemma's hash to figure out that actually these hashes came from different strings.
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    yield prev

            # Output
            prev = curr

        if prev is not None:
            yield prev  # Yes, not doing a hash check means it is possible that this one decomposition has already been seen before.

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

    def _readers(self) -> list[ModestReader[Any,MorphyNetDerivation]]:
        return [_MorphyNetReader_Derivation(verbose=self._verbose)]


class _MorphyNetReader_Derivation(ModestReader[tuple[str,str,str,str,str,str],MorphyNetDerivation]):

    def __init__(self, verbose: bool=False):
        self._verbose = verbose

    def _generateRaw(self, path: Path):
        yield from iterateTsv(path, verbose=self._verbose)

    def _createWriter(self):
        return TsvWriter()

    def _parseRaw(self, raw, id: int):
        original, result, original_pos, result_pos, affix, affix_type = raw
        return MorphyNetDerivation(
            id=id,
            word=result,
            base=original,
            affix=affix,
            prefix_not_suffix=(affix_type == "prefix")
        )
