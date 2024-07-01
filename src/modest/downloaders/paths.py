import datasets  # Ensure that the user has installed this package and hence has a HF_HOME.

import os
from pathlib import Path
import time
from typing import List

PATH_HFHOME = Path(os.environ["HF_HOME"])
PATH_HFHOME_MODEST = PATH_HFHOME / "datasets" / "modest"
PATH_HFHOME_MODEST.mkdir(parents=False, exist_ok=True)


class PathManagement:

    @staticmethod
    def datasetCache(language: str, name: str) -> Path:
        if not language or not name:
            raise ValueError("Dataset must have a language and a name!")
        return PathManagement._extendHome([language, name])

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
