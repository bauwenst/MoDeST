from abc import abstractmethod, ABC
from typing import Optional
from enum import Enum
from dataclasses import dataclass

from .trivial import SequenceOfStrings
from ..interfaces.morphologies import WordDecompositionWithLexicalForm


class GreekPOS(Enum):
    NOUN        = 1
    ADJECTIVE   = 2
    VERB        = 3
    PARTICIPLE  = 4
    NUMERIC     = 5
    ARTICLE     = 6
    PRONOUN     = 7
    ADVERB      = 8
    EXCLAMATION = 9
    NAME        = 10
    CONNECTOR   = 11
    PREPOSITION = 12
    DIMINUTIVE  = 13
    SUPERLATIVE = 14
    GERUND      = 15
    PARTICLE    = 16


class GreekCase(Enum):
    NOMINATIVE = 1
    ACCUSATIVE = 2
    GENITIVE   = 3
    CLITIC     = 4  # Apparently "klitiki" is the name of the vocative case in Greek.
    # DATIVE   = 5  # Only ancient Greek


class GreekNumber(Enum):
    SINGULAR = 1
    PLURAL   = 2


class GreekGender(Enum):
    MASCULINE = 1
    FEMININE  = 2
    NEUTER    = 3


class GreekTense(Enum):
    NON_PAST = 1
    PAST     = 2


class GreekAspect(Enum):
    IMPERFECTIVE = 1
    PERFECTIVE   = 2


class GreekMood(Enum):
    INDICATIVE = 1
    IMPERATIVE = 2
    INFINITIVE = 3

class GreekVoice(Enum):
    ACTIVE  = 1
    PASSIVE = 2

class GreekPerson(Enum):
    FIRST  = 1
    SECOND = 2
    THIRD  = 3


class GreekAnnotation(ABC):
    @abstractmethod
    def stringify(self):
        pass


@dataclass
class GreekEmptyAnnotation(GreekAnnotation):
    def stringify(self):
        return ""


@dataclass
class GreekNounAnnotation(GreekAnnotation):  # Nom. Masc. Sing.
    case:   GreekCase
    gender: GreekGender
    number: GreekNumber

    def stringify(self):
        return "|".join([self.case.name, self.gender.name, self.number.name])


@dataclass
class GreekVerbAnnotation(GreekAnnotation):  # Act. ind. pr. 1st sing.
    voice: GreekVoice
    mood: GreekMood

    # Compound tenses like "praesens", "imperfectum", ... are decomposed into "complete past" etc.
    aspect: Optional[GreekAspect]
    tense: Optional[GreekTense]

    person: GreekPerson
    number: GreekNumber

    def stringify(self):
        return "|".join([self.voice.name, self.mood.name] + ([self.aspect.name, self.tense.name] if self.mood != GreekMood.IMPERATIVE else []) + [self.person.name, self.number.name])


class GreekAnnotatedWord(WordDecompositionWithLexicalForm):

    def __init__(self, id: int, word: str, lemma: str,
                 morphemes: SequenceOfStrings, morphs: SequenceOfStrings,
                 pos: GreekPOS, annotation: GreekAnnotation):
        super().__init__(
            id=id,
            word=word,
            lemma=lemma,
            tag=pos.name + ":" + annotation.stringify()
        )
        self.pos        = pos
        self.annotation = annotation
        self.morphemes  = tuple(morphemes)
        self.morphs     = tuple(morphs)

    def decompose(self) -> tuple[str, ...]:
        return self.morphemes

    def segment(self) -> tuple[str, ...]:
        return self.morphs