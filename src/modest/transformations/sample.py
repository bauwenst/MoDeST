from collections import defaultdict
from typing import Iterator, Any, TypeVar
from pathlib import Path
from abc import abstractmethod

import numpy.random as npr
from modest.interfaces.kernels import ModestKernel

from ..interfaces.datasets import ModestDataset, M, Languageish
from ..interfaces.morphologies import WordSegmentationWithLemma


class _ReducedModestDataset(ModestDataset[M]):

    def __init__(self, nested: ModestDataset[M], do_cache: bool=True):
        super().__init__()
        self._nested = nested
        self._do_cache = do_cache

    def getCollectionName(self) -> str:  # Only for the dataset identifier.
        return self._nested.getCollectionName() + "_" + self._getModificationName()

    @abstractmethod
    def _getModificationName(self) -> str:
        pass

    def _getCachePath(self) -> Path:  # Not the same as the usual implementation of the cache path.
        folder = self._nested._getCachePath() / self._getModificationName()
        folder.mkdir(exist_ok=True)
        return folder

    def _getLanguage(self) -> Languageish:
        return self._nested._getLanguage()

    def _kernels(self) -> list[ModestKernel[Any,M]]:
        return self._nested._kernels()

    def _files(self) -> list[Path]:
        if not self._do_cache:
            return self._nested._files()

        cache_folder = self._getCachePath()

        paths = []
        for filtered_iterator, kernel, original_path in self._getFilteredGenerators():
            modified_path = cache_folder / original_path.name  # Note that original_path can be either a folder or a file, we don't know.
            if not modified_path.exists():
                kernel.writeObjects(filtered_iterator, in_path=original_path, out_path=modified_path)
            paths.append(modified_path)
        return paths

    @abstractmethod
    def _resetFilter(self):
        pass

    @abstractmethod
    def _filter(self, iterator: Iterator[M]) -> Iterator[M]:
        pass

    def _getFilteredGenerators(self) -> Iterator[tuple[Iterator[M], ModestKernel, Path]]:
        """
        The implementation of this method should resemble that of super().generate() as closely as possible.
        """
        self._resetFilter()
        for kernel, path in zip(self._kernels(), self._nested._files()):
            yield self._filter(kernel.generateObjects(path)), kernel, path

    def generate(self) -> Iterator[M]:
        if self._do_cache:  # zips the kernels and the cache paths generated above.
            yield from super().generate()
        else:  # You have to filter on-the-spot.
            for filtered_iterator, _, _ in self._getFilteredGenerators():
                yield from filtered_iterator


class _ElementwiseFilteredModestDataset(_ReducedModestDataset[M]):

    @abstractmethod
    def _keep(self, item: M) -> bool:
        pass

    @abstractmethod
    def _stop(self, item: M) -> bool:
        pass

    def _filter(self, iterator: Iterator[M]) -> Iterator[M]:
        for thing in iterator:
            if self._keep(thing):
                yield thing
            if self._stop(thing):
                break


class TruncateModestDataset(_ElementwiseFilteredModestDataset[M]):
    def __init__(self, nested: ModestDataset[M], desired_size: int, do_cache: bool=True):
        super().__init__(nested=nested, do_cache=do_cache)
        self._size = desired_size
        self._so_far = 0

    def _resetFilter(self):
        self._so_far = 0

    def _keep(self, item: M):
        return True

    def _stop(self, item: M) -> bool:
        self._so_far += 1
        return self._so_far >= self._size


class MinimumMorphemesModestDataset(_ElementwiseFilteredModestDataset[M]):
    def __init__(self, nested: ModestDataset[M], n_morphemes_minimum: int, do_cache: bool=True):
        super().__init__(nested=nested, do_cache=do_cache)
        self._n_morphemes_minimum = n_morphemes_minimum

    def _resetFilter(self):
        pass

    def _keep(self, item: M):
        return self._n_morphemes_minimum <= 1 or len(item.segment()) >= self._n_morphemes_minimum

    def _stop(self, item: M) -> bool:
        return False


class DropoutModestDataset(_ElementwiseFilteredModestDataset[M]):

    def __init__(self, nested: ModestDataset[M], desired_size: int, seed: int=0, do_cache: bool=True):
        super().__init__(nested=nested, do_cache=do_cache)

        self._size    = desired_size
        self._P_admit = min(1.0, desired_size / nested.card().size)  # Admit old examples with this probability.
        self._seed    = seed

        self._so_far  = 0
        self._rng     = npr.default_rng(seed=self._seed)
        assert self._P_admit <= 1

    def _getModificationName(self) -> str:
        return f"dropout-{self._size}_{self._seed}"

    def _resetFilter(self):
        self._so_far = 0
        self._rng    = npr.default_rng(seed=self._seed)

    def _keep(self, item: M) -> bool:
        keep = self._rng.random() < self._P_admit
        self._so_far += keep
        return keep

    def _stop(self, item: M) -> bool:
        return self._so_far >= self._size


T = TypeVar("T", bound=WordSegmentationWithLemma)

class SampleLexemes(_ReducedModestDataset[T]):
    """
    Finds lexemes for the given dataset (i.e. sets of segmentations with the same lemma),
    then samples N of them at random, and then samples one segmentation from each of them.
    """

    def __init__(self, nested: ModestDataset[T], n_lexemes: int, n_per_lexeme: int=1, seed: int=0, do_cache: bool=True):
        super().__init__(nested=nested, do_cache=do_cache)
        self._seed = seed
        self._n_lexemes    = n_lexemes
        self._n_per_lexeme = n_per_lexeme

    def _filter(self, iterator: Iterator[M]) -> Iterator[M]:  # You can't filter across kernels because when you write the cache, the IDs in the filter stream are cross-referenced with a per-kernel stream of raw examples.
        rng = npr.default_rng(seed=self._seed)

        id_to_object = {obj._id: obj for obj in iterator}  # Assumed to be unordered, so you just need to store the whole thing in memory.

        # Step 1: Find all lemmas.
        lemma_ids: dict[str,int]  = dict()
        lemma_id_to_object_ids: dict[int,list[int]] = defaultdict(list)
        for obj in id_to_object.values():
            lemma = obj.lemma
            if lemma not in lemma_ids:
                lemma_ids[lemma] = len(lemma_ids)
            lemma_id_to_object_ids[lemma_ids[lemma]].append(obj._id)
        del lemma_ids

        # Step 2: Choose lemmas and sample from their lexemes.
        for lemma_id in rng.choice(len(lemma_id_to_object_ids), size=self._n_lexemes, replace=False):
            ids_in_lexeme = lemma_id_to_object_ids[lemma_id]
            if len(ids_in_lexeme) > self._n_per_lexeme:
                ids_in_lexeme = rng.choice(ids_in_lexeme, size=self._n_per_lexeme, replace=False)
            for id in ids_in_lexeme:
                yield id_to_object[id]

        # # Step 3: Sample from lexemes. Assumption: all word forms of the same lexeme are grouped together in the dataset.
        # current_lemma  = None
        # current_lexeme = []
        # for obj in self._nested.generate():
        #     if current_lemma is None:
        #         current_lemma = obj.lemma
        #
        #     if obj.lemma != current_lemma:  # You have rolled into the next lexeme. Sample from the old one if it was relevant.
        #         if current_lexeme:
        #             lemma_ids.remove(current_lemma)
        #             yield current_lexeme[rng.integers(len(current_lexeme))]
        #
        #         current_lemma = obj.lemma
        #         current_lexeme = []
        #
        #     if current_lemma in lemma_ids:  # Only expand the current lexeme if relevant.
        #         current_lexeme.append(obj)
        #
        # if current_lexeme:  # End-of-loop cleanup
        #     yield current_lexeme[rng.integers(len(current_lexeme))]
