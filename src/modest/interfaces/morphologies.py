# TODO: I wonder if you can train an algorithm/model to recognise morphemes unsupervised based on
#       the supervised data of a word's (surface form, lexical form == lemma + tag) pair.
#       Having that tag should basically resolve any ambiguity.
from typing import Tuple
from abc import abstractmethod, ABC

from ..algorithms.alignment import alignMorphemes_Viterbi


class WordSegmentation(ABC):
    """
    For datasets that contain at least
        segmentation: meta/stas/ize
    """

    def __init__(self, word: str):
        self.word = word  # Also called the "surface form" a.o.t. the "lexical form" which is (lemma, tag). https://en.wikipedia.org/wiki/Morphological_dictionary

    @abstractmethod
    def segment(self) -> Tuple[str, ...]:  # Should be faster than a list.  https://stackoverflow.com/a/22140115/9352077
        pass


class WordDecomposition(WordSegmentation):  # All decompositions should be able to be segmented.
    """
    For datasets that contain at least
                 word: metastasize
        decomposition: meta stasis ize
    """

    @abstractmethod
    def decompose(self) -> Tuple[str, ...]:
        pass

    def segment(self) -> Tuple[str, ...]:
        return tuple(alignMorphemes_Viterbi(self.word, self.decompose())[0].split(" "))


class LexicalForm(ABC):
    """
    For datasets that contain at least
         word: reconstituées
        lemma: reconstituer
          tag: V|V.PTCP;PST|FEM|PL
    """

    def __init__(self, word: str, lemma: str, tag: str):
        self.word  = word
        self.lemma = lemma
        self.tag   = tag


class WordSegmentationWithLexicalForm(LexicalForm, WordSegmentation):
    """
    For datasets that contain at least
        segmentation: reconstitu|é|e|s
               lemma: reconstituer
                 tag: V|V.PTCP;PST|FEM|PL
    """
    pass


class WordDecompositionWithLexicalForm(LexicalForm, WordDecomposition):
    """
    For datasets that contain at least
                 word: reconstituées
        decomposition: reconstituer|é|e|s
                lemma: reconstituer
                  tag: V|V.PTCP;PST|FEM|PL

    """
    pass


class WordSegmentationWithFreeSegmentation(WordSegmentation):
    """
    For datasets that have both bound and free morphemes. The extra method provided gives a morphological segmentation
    where all bound morphemes have been concatenated to whatever morpheme they are bound to, so that all morphemes in
    the results are free morphemes. (The only exception are interfices, which, since they belong to two morphemes on
    either side, are allowed to appear separately.)

    This mode of segmentation is called "lexemic" in the thesis at https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf
    because free morphemes each have their own lexeme whereas bound morphemes don't, and it is called "whole-word" in
    the paper at https://aclanthology.org/2024.naacl-long.324/ (where it is also explained) because bound morphemes are
    not whole words whilst free morphemes are. One might call it "compounding-only segmentation" too.
    """

    @abstractmethod
    def segmentFree(self) -> Tuple[str, ...]:
        pass


class WordDecompositionWithFreeSegmentation(WordDecomposition):
    """
    CELEX is an example of this.

    Note that a "free-morpheme decomposition" also exists, see Minixhofer's CompoundPiece dataset.
    """

    @abstractmethod
    def segmentFree(self) -> Tuple[str, ...]:
        pass


##############################################################################################


class MorphologyVisitor(ABC):
    """
    In many tests in the code base, we want to have ONE procedure where a method of the above class
    is called but should be readily interchangeable. Without inheritance, this is possible just by
    passing the method itself as an argument and calling that on an object of the class (e.g. pass in
    method = LemmaMorphology.morphsplit and then call method(obj), which is equivalent to obj.morphSplit()).

    With inheritance, however, Python won't use the overridden version of the method dynamically, so all that is
    executed is the 'pass' body. The solution is a visitor design pattern.
    """
    @abstractmethod
    def __call__(self, morphology: WordSegmentation) -> Tuple[str, ...]:
        pass


class MorphSplit(MorphologyVisitor):
    def __call__(self, morphology: WordSegmentation) -> Tuple[str, ...]:
        return morphology.segment()


class MorphemeSplit(MorphologyVisitor):
    def __call__(self, morphology: WordDecomposition) -> Tuple[str, ...]:
        return morphology.decompose()


class FreeMorphSplit(MorphologyVisitor):
    def __call__(self, morphology: WordSegmentationWithFreeSegmentation) -> Tuple[str, ...]:
        return morphology.segmentFree()
