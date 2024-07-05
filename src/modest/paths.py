import datasets  # Ensure that the user has installed this package and hence has a HF_HOME.

from pathlib import Path
from typing import List
from langcodes import Language

import os
import time

PATH_HFHOME = Path(os.environ["HF_HOME"])
PATH_HFHOME_MODEST = PATH_HFHOME / "datasets" / "modest"
PATH_HFHOME_MODEST.mkdir(parents=False, exist_ok=True)


class PathManagement:

    @staticmethod
    def datasetCache(language: Language, dataset_name: str) -> Path:
        if not language.is_valid() or not dataset_name:
            raise ValueError("Dataset must have a valid language and a non-empty name!")
        return PathManagement._extendHome([language.language_name(), dataset_name])

    @staticmethod
    def namelessCache() -> Path:
        return PathManagement._extendHome(["_cache"])

    @staticmethod
    def makeTempFolder() -> Path:
        """
        Generates a throwaway folder that can later be deleted.
        """
        return PathManagement._extendHome(["_to-be-deleted-" + time.strftime("%Y%m%d-%H%M%S")])

    @staticmethod
    def _extendHome(parts: List[str]) -> Path:
        folder = PATH_HFHOME_MODEST
        for part in parts:
            folder /= part
        folder.mkdir(parents=True, exist_ok=True)
        return folder
