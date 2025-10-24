"""
A reader in MoDeST is a class that reads a file or folder of files with exactly one storage and tag format.
A dataset provides a unified interface overtop one or more readers.

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


class ModestReader(Generic[Raw,M], ABC):

    # Reading

    @abstractmethod
    def _generateRaw(self, path: Path) -> Iterator[Raw]:
        """
        Extracts information to be parsed for each example.
        If the path is required to be a folder, it is assumed that this function can find its way in that folder.
        """
        pass

    def generateRaw(self, path: Path) -> Iterator[tuple[int, Raw]]:
        yield from enumerate(self._generateRaw(path))

    @abstractmethod
    def _parseRaw(self, raw: Raw, id: int) -> M:
        """
        Parses one example. If not possible, raise an exception (rather than returning None).
        """
        pass

    def generateObjects(self, path: Path) -> Iterator[M]:
        for i, raw in self.generateRaw(path):
            try:
                yield self._parseRaw(raw, i)
            except Exception:  # Importantly, doesn't catch GeneratorExit.
                logger.info(f"Unparseable example: {raw}")

    # Writing

    @abstractmethod
    def _createWriter(self) -> "Writer[Raw]":
        """
        Implement if you want to support writing. Otherwise, raise NotImplementedError.
        """
        pass

    def writeObjects(self, objects: Iterator[M], in_path: Path, out_path: Path):
        write_stream = self._createWriter().openStream(out_path)
        read_stream = self.generateRaw(in_path)
        for obj in objects:
            found_raw = False
            looked_everywhere = False
            while not found_raw:
                for i, raw in read_stream:
                    if i == obj._id:  # This raw is the raw belonging to the object.
                        write_stream.send(raw)
                        found_raw = True
                        break
                else:
                    if looked_everywhere:
                        raise RuntimeError(f"Cannot find object id {obj._id} in the raw examples generate by the {self.__class__.__name__} reader.")
                    else:
                        read_stream = self.generateRaw(in_path)
                        looked_everywhere = True
        write_stream.close()


class Writer(Generic[Raw], ABC):
    """
    Manages all writes of raw versions of morphological objects to a file or folder.
    The output path is only known after construction.

    There are two ways to implement this:
        1. Using Python's 'with' context manager.
        2. With a generator coroutine, i.e. a single method
            def openStream(out_path: Path) -> Generator[None, Raw, None]:
                ...
        which handles its own setup (rather than having to implement an __enter__ and __exit__)
        and has to use a line like 'next_example = yield' which receives examples from .send().

    The first is difficult to implement but easy to use:

        with Writer(out_path) as writer:
            for raw in iterator:
                writer.write(raw)

    The second is easy to implement but difficult to use (i.e. you need to know more):

        stream = Writer().openStream(out_path)
        for raw in iterator:
            stream.send(raw)
        stream.close()

    But since we are the user and we hide this usage below caching methods, I will go for the easier implementation.
    """

    @abstractmethod
    def _createStream(self, out_path: Path) -> Generator[None, Raw, None]:
        pass

    def openStream(self, out_path: Path) -> Generator[None, Raw, None]:
        stream = self._createStream(out_path)
        stream.send(None)
        return stream
