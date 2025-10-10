"""
A kernel in MoDeST is a class that reads a file or folder of files with exactly one storage and tag format.
A dataset provides a unified interface overtop one or more kernels.

The reason for separating these two concepts is so that when composing two datasets, the default assumption is that
the datasets themselves already work with multiple formats and thus no extra complexity is introduced by suddenly
having more than one thing rather than one thing to iterate internally.
"""
from typing import TypeVar, Generic, Iterator, Generator, Self
from pathlib import Path
from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(__name__)

from .morphologies import _IdentifiedWord

Raw = TypeVar("Raw")
M = TypeVar("M", bound=_IdentifiedWord)


class ModestKernel(Generic[Raw,M], ABC):

    # Reading

    @abstractmethod
    def _generateRaw(self, path: Path) -> Iterator[tuple[int,Raw]]:
        """
        Extracts information to be parsed for each example.
        If the path is required to be a folder, it is assumed that this function can find its way in that folder.
        """
        pass

    @abstractmethod
    def _parseRaw(self, raw: Raw, id: int) -> M:
        """
        Parses one example. If not possible, raise an exception (rather than returning None).
        """
        pass

    def _generateObjects(self, path: Path) -> Iterator[M]:
        for i, raw in self._generateRaw(path):
            try:
                yield self._parseRaw(raw, i)
            except:
                logger.info(f"Unparseable example: {raw}")

    # Writing

    @abstractmethod
    def _createWriter(self, path: Path) -> "Writer[Raw]":
        """
        Implement if you want to support writing. Otherwise, raise NotImplementedError.
        """
        pass

    def _writeObjects(self, objects: Iterator[M], in_path: Path, out_path: Path):
        with self._createWriter(out_path) as writer:
            raws = self._generateRaw(in_path)
            for obj in objects:
                found_raw = False
                looked_everywhere = False
                while not found_raw:
                    for i, raw in raws:
                        if i == obj._id:  # This raw is the raw belonging to the object.
                            writer.write(raw)
                            found_raw = True
                            break
                    else:
                        if looked_everywhere:
                            raise RuntimeError(f"Cannot find object id {obj._id} in the raw examples generate by the {self._kernel.__class__.__name__} kernel.")
                        else:
                            raws = self._generateRaw(in_path)
                            looked_everywhere = True


class Writer(Generic[Raw], ABC):
    """
    Manages all writes of raw versions of morphological objects to a file or folder,
    using Python's 'with' context manager.

    An alternative way to implement this would be to have a single method
        def openStream(out_path: Path) -> Generator[None, Raw, None]:
            ...
    which would handle its own setup (rather than having to implement an __enter__ and __exit__)
    and would have to use a line like 'next_example = yield' which would receive examples from .send().
    So then as the user, rather than

        with Writer(out_path) as writer:
            for raw in iterator:
                writer.write(raw)

    you would have

        stream = Writer().openStream(out_path)
        for raw in iterator:
            stream.send(raw)
        stream.close()
    """

    def __init__(self, path: Path):
        self._out_path = path

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def write(self, raw: Raw):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
