"""
Object-oriented model for morphologically split lemmas.

Contains a general interface for any dataset format, and an
implementation specific to CELEX/e-Lex; the morphSplit algorithm
is more general though, and can be repurposed for e.g. Morpho Challenge datasets.
"""
from typing import Iterable
from abc import ABC, abstractmethod
from pathlib import Path


### INTERFACE ###
class LemmaMorphology(ABC):

    @abstractmethod
    def lemma(self) -> str:
        pass

    @abstractmethod
    def morphSplit(self) -> str:
        pass

    @abstractmethod
    def lexemeSplit(self) -> str:
        pass

    @staticmethod  # Can't make it abstract, but you should implement this.
    def generator(file: Path) -> Iterable["LemmaMorphology"]:
        """
        Generator to be used by every script that needs morphological objects.
        """
        raise NotImplementedError()


### VISITORS ###
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
    def __call__(self, morphology: LemmaMorphology):
        pass


class MorphSplit(MorphologyVisitor):
    def __call__(self, morphology: LemmaMorphology):
        return morphology.morphSplit()


class LexSplit(MorphologyVisitor):
    def __call__(self, morphology: LemmaMorphology):
        return morphology.lexemeSplit()
