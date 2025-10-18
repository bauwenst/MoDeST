from pathlib import Path
import shutil
import requests

from ..interfaces.datasets import Languageish
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection, MorphynetSubset, MORPHYNET_LANGUAGES


class Russian_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def _getLanguage(self) -> Languageish:
        return "Russian"


class Russian_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def _getLanguage(self) -> Languageish:
        return "Russian"

    def _files(self) -> list[Path]:
        morphynet_code = MORPHYNET_LANGUAGES.get(self.getLanguage())
        cache_path = self._getCachePath() / f"{morphynet_code}.{self.getSubset().toString()}.v1.tsv"
        if not cache_path.exists():
            # Download ZIP file.
            zip_path = cache_path.with_suffix(".zip")

            ###
            url = f"https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{morphynet_code}/{self._getRemoteFilename(morphynet_code, self.getSubset())}"
            response = requests.get(url)
            with open(zip_path, "wb") as handle:
                handle.write(response.content)
            ###

            # Extract ZIP file, delete it, and assert that the TSV exists.
            shutil.unpack_archive(zip_path, cache_path.parent)
            zip_path.unlink()
            assert cache_path.exists()

        return [cache_path]

    def _getRemoteFilename(self, langcode: str, subset: MorphynetSubset) -> str:
        return "rus.inflectional.v1.zip"
