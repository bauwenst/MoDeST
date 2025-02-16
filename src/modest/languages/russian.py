from pathlib import Path
import langcodes
import shutil

from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection, MorphynetSubset, MORPHYNET_LANGUAGES


class Russian_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def __init__(self):
        super().__init__(language=langcodes.find("Russian"))


class Russian_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def __init__(self):
        super().__init__(language=langcodes.find("Russian"))

    def _get(self) -> Path:
        morphynet_code = MORPHYNET_LANGUAGES.get(self._language)
        cache_path = self._getCachePath() / f"{morphynet_code}.{self._subset.toString()}.v1.tsv"
        if not cache_path.exists():
            # Download ZIP file.
            zip_path = cache_path.with_suffix(".zip")

            ###
            url = f"https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/{morphynet_code}/{self._getRemoteFilename(morphynet_code, self._subset)}"
            response = requests.get(url)
            with open(zip_path, "wb") as handle:
                handle.write(response.content)
            ###

            # Extract ZIP file, delete it, and assert that the TSV exists.
            shutil.unpack_archive(zip_path, cache_path.parent)
            zip_path.unlink()
            assert cache_path.exists()

        return cache_path

    def _getRemoteFilename(self, langcode: str, subset: MorphynetSubset) -> str:
        return "rus.inflectional.v1.zip"
