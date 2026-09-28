"""
For downloading data from WebCelex.
"""
from pathlib import Path
from typing import Any

from tktkt.util.types import L

from ..formats.celex import CelexLemmaMorphology
from ..formats.tsv import iterateHandle, iterateTsv
from ..interfaces.datasets import ModestDataset
from ..interfaces.readers import ModestReader

CELEX_LANGUAGES = {
    L("English"): "English",
    L("German"): "German",
    L("Dutch"): "Dutch"
}

_WEBCELEX_NEEDS_AUTHENTICATION = True  # It used not to :(


class CelexDataset(ModestDataset[CelexLemmaMorphology]):

    def __init__(self, verbose: bool=False, legacy: bool=False):
        super().__init__()
        self._verbose = verbose
        self._legacy = legacy

    def getCollectionName(self) -> str:
        return "CELEX"

    def _readers(self) -> list[ModestReader[tuple[str,str],CelexLemmaMorphology]]:
        return [_CelexReader(self._verbose, self._legacy)]

    def _files(self) -> list[Path]:
        full_name = CELEX_LANGUAGES.get(self.getLanguage())
        if full_name is None:
            raise ValueError(f"Unknown language: {self.getLanguage()}")

        cache_path = self._getCachePath() / (f"{self.getLanguage().to_tag()}.struclab.tsv")
        if not cache_path.exists():
            if not _WEBCELEX_NEEDS_AUTHENTICATION:
                from ..util.web import getWebDriver, By, time, WebDriverWait, bs4

                print("Simulating browser to download CELEX dataset (takes under 60 seconds)...")
                driver = getWebDriver()

                # Entry page to select language
                try:
                    driver.get("http://celex.mpi.nl/scripts/entry.pl")
                except Exception as e:
                    print("Oh oh! It seems that CELEX is dead :(")
                    raise e
                driver.find_element(by=By.LINK_TEXT, value=f"{full_name} Lemmas").click()

                # Select column
                driver.switch_to.frame(driver.find_element(by=By.CSS_SELECTOR, value=f"frame[name='{full_name.lower()}_lemmas_cols']"))
                driver.find_element(by=By.CSS_SELECTOR, value="option[value='StrucLab']").click()
                driver.find_element(by=By.CSS_SELECTOR, value="input[type='submit']").click()

                # Skip constraints
                driver.switch_to.default_content()
                driver.find_element(by=By.CSS_SELECTOR, value="input[type='submit']").click()

                # Get tabular format and add word surface forms
                driver.find_element(by=By.CSS_SELECTOR, value="input[name='word']").click()
                driver.find_element(by=By.CSS_SELECTOR, value="input[name='fixit']").click()
                driver.find_element(by=By.CSS_SELECTOR, value="input[type='submit']").click()

                # Wait until the final page appears.
                time.sleep(3)

                # Wait until that page has loaded. It gets 60 seconds to do so.  https://stackoverflow.com/a/30385843/9352077
                WebDriverWait(driver, timeout=60).until(lambda d: d.execute_script("return document.readyState") == "complete")

                # Parse the page and turn it into a TSV.
                table = driver.find_element(by=By.TAG_NAME, value="table")
                soup = bs4.BeautifulSoup(table.get_attribute("outerHTML"), features="lxml")
                with open(cache_path, "w", encoding="utf-8") as out_handle:
                    first = True
                    for row in soup.find("table").find("tbody").find_all("tr"):
                        if first:
                            first = False
                            continue

                        word = row.find("td")
                        tag = word.find_next("td")
                        if tag.text:
                            out_handle.write(word.text + "\t" + tag.text + "\n")
                print("Successfully downloaded CELEX.")
            else:
                import requests

                url = f"https://raw.githubusercontent.com/bauwenst/MoDeST/master/data/in/webcelex/celex_morphology_{self.getLanguage().to_tag()}.txt"
                response = requests.get(url)
                if response.status_code == 404:
                    raise RuntimeError(f"The URL {url} does not exist.")
                with open(cache_path, "wb") as handle:
                    handle.write(response.content)

        return [cache_path]

    def _cleanFile(self, file: Path):
        """
        Removes lines that do not conform to the {spaceless string}\t{spaceless string} format.
        """
        with open(file.with_stem(file.stem + "_proper"), "w", encoding="utf-8") as out_handle:
            with open(file, "r", encoding="utf-8") as in_handle:
                for line in iterateHandle(in_handle):
                    parts = line.split("\t")
                    if len(parts) == 2 and " " not in line:
                        out_handle.write(line + "\n")


class _CelexReader(ModestReader[tuple[str,str],CelexLemmaMorphology]):

    def __init__(self, verbose: bool, legacy: bool):
        self._verbose= verbose
        self._legacy = legacy

    def _generateRaw(self, path: Path):
        yield from iterateTsv(path, verbose=self._verbose)

    def _parseRaw(self, raw: tuple[str,str], id: int):
        """
        TODO: From what I can guess (there is no manual for CELEX tags!), the [F] tag is used to indicate participles
              (past and present), which are treated as a single morpheme even though they clearly are not. For some,
              you can deduce the decomposition by re-using the verb's decomposition, so you could write some kind of
              a dataset sanitiser for that.
        """
        word, tag = raw
        if "[F]" not in tag and (self._legacy or "'" not in word):
            return CelexLemmaMorphology(id=id, lemma=word, celex_struclab=tag)
        else:
            raise

    def _createWriter(self):
        raise NotImplementedError()
