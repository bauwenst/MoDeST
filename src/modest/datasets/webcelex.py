"""
For downloading data from WebCelex.
"""
import langcodes
from pathlib import Path
from typing import Iterable

# Web stuff
# - Make browser
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# - Parse stuff in browser
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import time
import bs4

from ..formats.celex import CelexLemmaMorphology
from ..formats.tsv import iterateHandle
from ..interfaces.datasets import ModestDataset
from ..paths import PathManagement


CELEX_LANGUAGES = {
    langcodes.find("English"): "English",
    langcodes.find("German"): "German",
    langcodes.find("Dutch"): "Dutch"
}


class CelexDataset(ModestDataset):

    def __init__(self, language: langcodes.Language):
        self.language = language

    def _get(self) -> Path:
        full_name = CELEX_LANGUAGES.get(self.language)
        if full_name is None:
            raise ValueError(f"Unknown language: {self.language}")

        cache_path = PathManagement.datasetCache(language=self.language, dataset_name="CELEX") / (f"{self.language.to_tag()}.struclab.tsv")
        if not cache_path.exists():
            chrome_options = Options()
            chrome_options.headless = True
            # chrome_options.add_experimental_option("detach", True)  # Add this if you want the browser to stay open after the experiment is done.
            driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)  # If you get a "ValueError: There is no such driver by url", you need to pip upgrade the webdriver_manager package.
            driver.implicitly_wait(3)  # Waiting time in case an element isn't found (WebCelex is slow...).

            # Entry page to select language
            driver.get("http://celex.mpi.nl/scripts/entry.pl")
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

        return cache_path

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

    def _generator(self, file: Path, verbose=True, legacy=False) -> Iterable[CelexLemmaMorphology]:
        # FIXME: This is probably a parser for MY specific format for CELEX, but this might not be how you download it
        #        from WebCelex.
        with open(file, "r", encoding="utf-8") as handle:
            for line in iterateHandle(handle, verbose=verbose):
                lemma, morphological_tag = line.split("\t")
                try:
                    if "[F]" not in morphological_tag and (legacy or "'" not in lemma):  # TODO: From what I can guess (there is no manual for CELEX tags!), the [F] tag is used to indicate participles (past and present), which are treated as a single morpheme even though they clearly are not. For some, you can deduce the decomposition by re-using the verb's decomposition, so you could write some kind of a dataset sanitiser for that.
                        yield CelexLemmaMorphology(lemma=lemma, celex_struclab=morphological_tag)
                except:
                    print(f"Failed to parse morphology: '{lemma}' tagged as '{morphological_tag}'")
