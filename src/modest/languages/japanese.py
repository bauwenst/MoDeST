from pathlib import Path
from datasets import DatasetDict

import re

from modest.interfaces.datasets import Languageish
from modest.datasets.ud import UniversalDependencies


class JapaneseUD(UniversalDependencies):

    def _getLanguage(self) -> Languageish:
        return "Japanese"

    def _subset(self) -> str:
        return "ja_gsd"

    def _datasetToFiles(self, input_dataset: DatasetDict, output_folder: Path, output_stem: str= "ud") -> list[Path]:
        pattern = re.compile(r"LUWBILabel=([BI])")

        files = []
        for split in input_dataset:
            # Generate deduplicated segmentations.
            segmentations = dict()
            for thing in input_dataset[split]:
                span_starts = [i for i, tag in enumerate(thing["misc"]) if pattern.search(tag).group(1) == "B"]
                suws = [thing["tokens"][start:end] for start, end in zip(span_starts, span_starts[1:])]
                for segmentation in suws:
                    segmentations["".join(segmentation)] = tuple(segmentation)

            # Store segmentations in a TSV file.
            file = output_folder / (output_stem + "-" + split + ".tsv")
            with open(file, "w", encoding="utf-8") as handle:
                for word in sorted(segmentations):
                    handle.write(f"{word}\t{' '.join(segmentations[word])}\n")
            files.append(file)

        return files
