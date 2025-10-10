from typing import Tuple
from abc import abstractmethod, ABC

from ..algorithms.alignment import alignMorphemes_Viterbi


class _IdentifiedWord:
    def __init__(self, id: int, word: str):
        self._id = id
        self.word = word  # Also called the "surface form" a.o.t. the "lexical form" which is (lemma, tag). https://en.wikipedia.org/wiki/Morphological_dictionary


class _AddSegmentation(ABC):
    @abstractmethod
    def segment(self) -> Tuple[str, ...]:  # Should be faster than a list.  https://stackoverflow.com/a/22140115/9352077
        pass


class WordSegmentation(_IdentifiedWord, _AddSegmentation):
    """
    For datasets that contain at least
        segmentation: meta/stas/ize
    """
    pass


class _AddDecomposition(ABC):
    @abstractmethod
    def decompose(self) -> Tuple[str, ...]:
        pass


class WordDecomposition(WordSegmentation, _AddDecomposition):  # All decompositions should be able to be segmented.
    """
    For datasets that contain at least
                 word: metastasize
        decomposition: meta stasis ize
    """

    def segment(self) -> Tuple[str, ...]:
        return tuple(alignMorphemes_Viterbi(self.word, self.decompose())[0].split(" "))


class _IdentifiedLexicalForm(_IdentifiedWord):
    """
    For datasets that contain at least
         word: reconstituées
        lemma: reconstituer
          tag: V|V.PTCP;PST|FEM|PL
    """

    def __init__(self, id: int, word: str, lemma: str, tag: str):
        super().__init__(id, word)
        self.lemma = lemma
        self.tag   = tag


class WordSegmentationWithLexicalForm(_IdentifiedLexicalForm, WordSegmentation):
    """
    For datasets that contain at least
        segmentation: reconstitu|é|e|s
               lemma: reconstituer
                 tag: V|V.PTCP;PST|FEM|PL
    """
    pass


class WordDecompositionWithLexicalForm(WordSegmentationWithLexicalForm, _AddDecomposition):  # TODO: The one downside of this kind of inheritance is that this class is not a subclass of WordDecomposition.
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

    Note that a "free-morpheme decomposition" (which would have a method decomposeFree) also exists,
    see Minixhofer's CompoundPiece dataset.
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

    With inheritance, however, when you pass ParentClass.methodName to a function (because you don't know which subclass
    will be used, only that you want to call methodName on it), Python won't use the overridden version SubClass.methodName
    dynamically, so all that is executed is the 'pass' body. The solution is a visitor design pattern.
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
