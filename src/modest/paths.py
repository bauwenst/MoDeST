from datasets.config import HF_DATASETS_CACHE  # Ensure that the user has installed this package and hence has a HF_HOME.

from pathlib import Path
from langcodes import Language

import time

PATH_HFHOME_MODEST = HF_DATASETS_CACHE / "modest"
PATH_HFHOME_MODEST.mkdir(parents=True, exist_ok=True)


class PathManagement:

    @staticmethod
    def datasetCache(language: Language, dataset_name: str) -> Path:
        """
        Returns a folder in which to store data-specific files for future use.
        """
        if not language.is_valid() or not dataset_name:
            raise ValueError("Dataset must have a valid language and a non-empty name!")
        return PathManagement._extendHome([language.language_name(), dataset_name])

    @staticmethod
    def namelessCache() -> Path:
        """
        Returns a folder shared by all datasets.
        Will not be deleted, but you should not store dataset-specific data here.
        """
        return PathManagement._extendHome(["_cache"])

    @staticmethod
    def makeTempFolder() -> Path:
        """
        Generates a throwaway folder that can later be deleted.
        """
        return PathManagement._extendHome(["_to-be-deleted-" + time.strftime("%Y%m%d-%H%M%S")])

    @staticmethod
    def _extendHome(parts: list[str]) -> Path:
        folder = PATH_HFHOME_MODEST
        for part in parts:
            folder /= part
        folder.mkdir(parents=True, exist_ok=True)
        return folder
