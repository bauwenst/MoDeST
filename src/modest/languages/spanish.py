from pathlib import Path
import requests

from ..interfaces.datasets import Languageish
from ..datasets.morphynet import MorphyNetDataset_Derivation, MorphyNetDataset_Inflection


class Spanish_MorphyNet_Derivations(MorphyNetDataset_Derivation):
    def _getLanguage(self) -> Languageish:
        return "Spanish"


class Spanish_MorphyNet_Inflections(MorphyNetDataset_Inflection):
    def _getLanguage(self) -> Languageish:
        return "Spanish"

    def _get(self) -> Path:
        cache_path = self._getCachePath() / "spa.inflectional.v1.tsv"
        if not cache_path.exists():
            url1 = f"https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/spa/spa.inflectional.v1.part1.tsv"
            url2 = f"https://raw.githubusercontent.com/kbatsuren/MorphyNet/main/spa/spa.inflectional.v1.part2.tsv"
            response1 = requests.get(url1)
            response2 = requests.get(url2)
            with open(cache_path, "wb") as handle:
                handle.write(response1.content)
                handle.write("\n".encode("utf-8"))
                handle.write(response2.content)

        return cache_path
