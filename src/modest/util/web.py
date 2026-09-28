from pathlib import Path
import os

# - To make browser
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# - To parse stuff in browser (importable with *)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import time
import bs4


__all__ = ["getWebDriver", "bs4", "time", "WebDriverWait", "By"]


def getWebDriver(download_folder: Path=None) -> webdriver.Chrome:
    # Make options (custom arguments)
    chrome_options = Options()
    chrome_options.headless = True
    # chrome_options.add_experimental_option("detach", True)  # Add this if you want the browser to stay open after the experiment is done.
    if download_folder is not None:
        chrome_options.add_experimental_option('prefs', {'download.default_directory': download_folder.as_posix()})

    # Make service (mandatory arguments)
    if os.name == "nt":
        driver_path = Path(ChromeDriverManager().install()).parent / "chromedriver.exe"  # To prevent "OSError: [WinError 193] %1 is not a valid Win32 application". https://stackoverflow.com/a/78797164/9352077
    else:
        driver_path = Path(ChromeDriverManager().install())
    service = Service(executable_path=driver_path.as_posix())

    # Instantiate driver
    driver = webdriver.Chrome(service=service, options=chrome_options)  # If you get a "ValueError: There is no such driver by url", you need to pip upgrade the webdriver_manager package.
    driver.implicitly_wait(3)  # Waiting time in case an element isn't found (WebCelex is slow...).
    return driver
