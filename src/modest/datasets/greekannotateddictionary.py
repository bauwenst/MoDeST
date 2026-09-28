"""
Downloads a Modern Greek dataset from the European Language Grid:
https://live.european-language-grid.eu/catalogue/lcr/7360/overview/
"""
from typing import Iterator, Any
from pathlib import Path

import json
from zipfile import ZipFile
from logging import getLogger

from tktkt.util.iterables import streamProgress
from tktkt.preparation.mappers import FilterDiacritics

logger = getLogger(__name__)

from ..interfaces.readers import ModestReader
from ..interfaces.datasets import ModestDataset, Languageish
from ..formats.greekannotateddictionary import *
from ..util.web import *


class GreekAnnotatedDictionary(ModestDataset[GreekAnnotatedWord]):
    """
    FIXME: I have observed that a lot of words prefixed with "υπο" are not tagged with a prefix
           in the dataset. Probably there are more prefixes like this too.
           I should ask my Greek friends to go through these when I get a hold of them.
    """

    def __init__(self, verbose: bool=False):
        super().__init__()
        self._verbose = verbose

    def getCollectionName(self) -> str:
        return "GreekAnnotDict"

    def _getLanguage(self) -> Languageish:
        return "Greek"

    def _readers(self) -> list[ModestReader[Any,GreekAnnotatedWord]]:
        return [_GADReader(verbose=self._verbose)]

    def _files(self) -> list[Path]:
        cache = self._getCachePath()
        downloaded_file = cache / "dataset.zip"
        extracted_file  = cache / f"greek_annotated_dictionary.json"
        if not extracted_file.exists():
            if not downloaded_file.exists():
                print("Simulating browser to download GAD dataset (takes under 60 seconds)...")
                driver = getWebDriver(download_folder=cache.parent)

                try:
                    driver.get(f"https://live.european-language-grid.eu/catalogue/lcr/7360/download/")
                except Exception as e:
                    print("Oh oh! It seems that European Language Grid is dead :(")
                    raise e
                driver.find_element(by=By.ID, value="rcc-decline-button").click()
                driver.find_element(by=By.CLASS_NAME, value="MuiCardHeader-action").click()
                driver.find_element(by=By.CLASS_NAME, value="access_button").click()
                time.sleep(10)
                driver.quit()
                assert downloaded_file.exists()

            with ZipFile(downloaded_file, "r") as handle:
                handle.extractall(cache)

            assert extracted_file.exists()

        return [extracted_file]


def findLongestPrefixIn(short: str, long: str) -> Optional[tuple[int,int]]:
    """
    Given a long string and a short string, find matches of prefixes of the short string in the
    long string, and return the start and length of the longest prefix.
    """
    best_start  = None
    best_length = 0
    for i in range(len(long)):
        # Shortcut: if you find the exact string, you can immediately return it because there will be no longer match.
        if long.startswith(short, i):
            return i, len(short)

        # Else, truncate to a prefix longer than the best one found so far.
        for L in range(len(short)-1, best_length, -1):
            if long.startswith(short[:L], i):
                best_start  = i
                best_length = L
                break

    return (best_start, best_length) if best_start is not None else None


def isGreekPrefixPresent(prefix: str, word: str) -> bool:
    if prefix == "εν":
        return word.startswith(prefix) or word.startswith("εγ") or word.startswith("εμ")
    elif prefix == "αμφι":
        return word.startswith(prefix) or word.startswith("αμφε")
    elif prefix == "συν":
        return word.startswith(prefix) or word.startswith("συγ")
    else:
        return word.startswith(prefix)


class _GADReader(ModestReader[dict, GreekAnnotatedWord]):
    """
    Example noun:
        {
            "name":"σύμφωνες",
            "partOfSpeech":"adjective",
            "morphologicalInfo":{"suffix":"ες","suffixType":"visual"},
            "inflectionalInfo": [
                {
                    "gender":"feminine",
                    "inflectionSuffix":"ες",
                    "isMainLemma":false,
                    "number":"plural",
                    "stem":"σύμφωνη",
                    "case":"accusative"
                },
                {
                    "gender":"feminine",
                    "inflectionSuffix":"ες",
                    "isMainLemma":false,
                    "number":"plural",
                    "stem":"σύμφωνη",
                    "case":"clitic"
                },
                {
                    "gender":"feminine",
                    "inflectionSuffix":"ες",
                    "isMainLemma":false,
                    "number":"plural",
                    "stem":"σύμφωνη",
                    "case":"nominative"
                }
            ],
            "graphemePhoneme":["σ-s","ύ-ˈi","μ-m","φ-f","ω-o","ν-n","ε-e","ς-s"],
            "syllables":["σύμ","φω","νες"],
            "phonetic":"/sˈimfones/",
            "cvForm":"-cvc-cv-cvc-",
            "numberOfCharacters":8,
            "numberOfPhonemes":8,
            "numberOfSyllables":3,
            "additionalInfo":{"wordFrequency":203},
            "featureInfo":{"7":[[0,5]],"58":[[0,1]],"80":[[5,6]],"407":[[3,4]],"411":[[4,5]],"413":[[6,7]],"420":[[2,3]],"425":[[0,1]],"427":[[5,6]]},
        }

    Example verb:
        {
            "name":"εισχωρώ",
            "partOfSpeech":"verb",
            "morphologicalInfo":{
                "prefix":"εισ",
                "prefixType":"add",

                "suffix":"ώ",
                "suffixType":"visual"
            },
            "inflectionalInfo":[
                {
                    "aspect":"imperfective",
                    "inflectionSuffix":"ώ",
                    "isMainLemma":true,
                    "mood":"indicative",
                    "number":"singular",
                    "person":"1",
                    "tense":"non_past",
                    "voice":"active"
                }
            ]

            "phonetic":"/isxorˈo/",
            "graphemePhoneme":["ει-i","σ-s","χ-x","ω-o","ρ-r","ώ-ˈo"],
            "cvForm":"-vv-ccv-cv-",
            "syllables":["ει","σχω","ρώ"],
            "numberOfCharacters":7,
            "numberOfPhonemes":6,
            "numberOfSyllables":3,
            "additionalInfo":{"wordFrequency":251},
            "featureInfo":{"34":[[0,2]],"90":[[5,6]],"223":[[6,7]],"247":[[0,3]],"411":[[4,5],[6,7]],"413":[[0,1]],"416":[[3,4]],"424":[[5,6]],"425":[[2,3]]},
        }
    """

    def __init__(self, verbose: bool):
        self._verbose = verbose
        self._accents = FilterDiacritics()

    def _generateRaw(self, path: Path) -> Iterator[dict]:
        with open(path, "r", encoding="utf-8") as handle:
            d = json.load(handle)
            yield from d["lemmas"] if not self._verbose else streamProgress(d["lemmas"])

    def _parseRaw(self, raw: dict, id: int) -> GreekAnnotatedWord:
        word: str  = raw["name"]
        pos        = GreekPOS[raw["partOfSpeech"].upper()]
        inflection = raw["inflectionalInfo"][0]

        # First make an annotation.  FIXME: Not all classes are covered here.
        if pos == GreekPOS.NOUN or pos == GreekPOS.ADJECTIVE:
            try:
                annotation = GreekNounAnnotation(
                    case=GreekCase[inflection["case"].upper()],
                    gender=GreekGender.MASCULINE if inflection["gender"].startswith("masc") else GreekGender.FEMININE if inflection["gender"].startswith("fem") else GreekGender.NEUTER,
                    number=GreekNumber[inflection["number"].upper()]
                )
            except Exception as e:  # Most of these are either letter names (alpha, beta, ...) or loanwords (kompiouter, frikasé, festibal, trakter, ...).
                raise
        elif pos == GreekPOS.VERB:
            personmap = {
                "1": "first",
                "2": "second",
                "3": "third",
            }

            mood = GreekMood[inflection["mood"].upper()]
            if mood == GreekMood.IMPERATIVE:
                aspect, tense = None, None
            else:
                aspect = GreekAspect[inflection["aspect"].upper()]
                tense  = GreekTense[inflection["tense"].upper()]
            number = GreekNumber[inflection["number"].upper()]
            person = GreekPerson[personmap[inflection["person"]].upper()]
            voice  = GreekVoice[inflection["voice"].upper()]

            annotation = GreekVerbAnnotation(voice, mood, aspect, tense, person, number)
        else:
            annotation = GreekEmptyAnnotation()

        # Find lemma
        if not inflection["isMainLemma"]:
            lemma = inflection["stem"]
        elif "stem" in inflection and inflection["stem"] != raw["name"] and raw["partOfSpeech"] not in {"adjective", "pronoun", "article", "participle"}:
            # These are weird edge cases where it says "yeah this is the lemma, but also, it's not".
            raise
        else:
            lemma = word

        if word[0].isupper() or word[0].isdigit():
            raise

        # print(word)
        raw.pop("cvForm")
        raw.pop("featureInfo")
        raw.pop("phonetic")
        raw.pop("syllables")
        raw.pop("additionalInfo")
        raw.pop("numberOfCharacters")
        raw.pop("numberOfPhonemes")
        raw.pop("numberOfSyllables")
        raw.pop("graphemePhoneme")
        # print(raw)

        # The three sources of morphological information.
        m = raw["morphologicalInfo"]
        d = m.get("derivationalInfo", [dict()])[0]
        i = raw["inflectionalInfo"][0]
        has_prefix = False
        if "prefixType" in m:
            if m["prefixType"] == "drop":  # A case we are not interested in
                m.pop("prefixType")
                m.pop("prefix")
            else:
                has_prefix = isGreekPrefixPresent(m["prefix"], word)  # Can be false even when a prefix is given.

        if not d and not has_prefix and "inflectionSuffix" not in i:
            raise

        # - Prefixes:
        #       - Only appear in morphological info, which may be absent.
        #       - When they are present, they are morphemes, not morphs.
        #       - Can be derived from derivational info stem.
        # - Stems:
        #       - Either in derivational info or inflectional info.
        # - Suffixes:
        #       - Derivational info only has suffixes.
        #       - It never occurs that the morphological info has a suffix but the inflectional info doesn't have one, or has a different one.
        #       - When there is derivational information, we say it is preferred over the inflection.
        #       - Suffixes from derivation and inflection (and, for that reason, morphological info) are morphs.
        morphemes = []
        morphs    = []

        has_suffix = "suffix" in d or "inflectionSuffix" in i
        if has_suffix:  # First thing to do is pop off a potential suffix.
            if d:
                suffix_theoretical = d["suffix"]
            else:
                suffix_theoretical = i["inflectionSuffix"]

            assert self._accents.convert(word).endswith(suffix_theoretical) or word.replace("ΐ", "ί").replace("ΰ", "ύ").endswith(suffix_theoretical)  # Sometimes the suffix has no accents. You need to compare this to the word stripped of its accents.
            word, suffix_actual = word[:-len(suffix_theoretical)], word[-len(suffix_theoretical):]

        # We will pop it off here already, so that the prefix doesn't accidentally get recognised as stem.
        if has_prefix:
            prefix_theoretical = m["prefix"]
            prefix_actual, word = word[:len(prefix_theoretical)], word[len(prefix_theoretical):]

        has_stem = "stemName" in d or "stem" in i
        has_preamble = False
        if has_stem:
            if "stemName" in d:
                stem = d["stemName"]
            else:
                stem = i["stem"]

            # Sometimes, but not always, this stem contains the known prefix. Parse that out first to sanitise the stem.
            if has_prefix and isGreekPrefixPresent(self._accents.convert(m["prefix"]), self._accents.convert(stem)):
                stem = stem[len(m["prefix"]):]

            # Now we can parse out this stem and potentially find a string that precedes it, which will be mapped to the prefix.
            match = findLongestPrefixIn(self._accents.convert(stem), self._accents.convert(word))
            if match:  # Basically always the case except for some numerical adjectives where the stem and the suffix are the same, so it doesn't matter.
                start, length = match
                preamble, word = word[:start], word[start:]
                if preamble:  # Between prefix (if it exists) and stem
                    has_preamble = True
            else:  # These are mostly verbs with irregular past roots, e.g. τρώ/ω -> έ/φαγ/ε. Unfortunately, in these cases we don't get the root from the dataset and it's impossible to impute them. What you can do is to only consider the suffix.
                pass

        # Now gather all the morphs and morphemes.
        if has_prefix:
            morphs.append(prefix_actual)
            morphemes.append(prefix_theoretical)

        if has_preamble:
            morphs.append(preamble)
            morphemes.append(preamble)

        morphs.append(word)
        if has_stem:
            morphemes.append(stem)
        else:
            morphemes.append(word)

        if has_suffix:
            morphs.append(suffix_actual)
            morphemes.append(suffix_theoretical)

        assert "".join(morphs) == raw["name"]
        assert len(morphs) == len(morphemes), f"{morphs} -- {morphemes}"
        return GreekAnnotatedWord(id=id, word=raw["name"], lemma=lemma, pos=pos, annotation=annotation, morphs=morphs, morphemes=morphemes)

    def _createWriter(self):
        raise NotImplementedError()
