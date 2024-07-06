<img src="doc/logo.png">

# MoDeST: a Morphological Decomposition &amp; Segmentation Trove
The point of MoDeST is two-fold:
1. Provide a general object-oriented Python interface to access morphological decompositions and segmentations;
2. Host morphological datasets generated by smaller research groups that would otherwise have a hard time being found.

*Morphological decomposition* is the task of recognising which building blocks a word was originally constructed from. These building blocks are its *morphemes*.
As an example, the Dutch derivation `isometrisch` ("isometric") can be decomposed into the morphemes `iso`, `meter` and `isch`.

*Morphological segmentation* is the task of isolating the substrings of a word that correspond to its morphemes. These substrings are called *morphs*.
In the above example, the segmentation would be `iso/metr/ic`.

## Languages and Datasets
The supported languages are simply under `modest.languages`, so the list will not be reproduced here.
The list of datasets roughly coincides with the downloaders under `modest.datasets`. Currently, the package supports:
- CELEX
- MorphyNet
- MorphoChallenge2010
- CompoundPiece

## Installation
Run
```shell
pip install "modest[github] @ git+https://github.com/bauwenst/MoDeST.git"
```

## Repo layout
Currently, the repo looks as follows:
```
data/              ---> Datasets hosted specifically by MoDeST on GitHub. Will NOT be downloaded when you install the package.
src/modest/        ---> All source code for the Python package that will be installed in your interpreter.
    languages/     ---> Per-language definitions of the classes users will interact with.
    datasets/      ---> Support code for pulling in and reading remote data.
    formats/       ---> Support code for turning tag formats into objects. (Tag formats are independent of how the tags are stored.)
    interfaces/    ---> Declarations of the interfaces users will interact with.
```

Currently, every language has its own file under `languages/`. The assumption is that the datasets pertaining to one language 
are sufficiently encapsulated that this will not clutter the imports from such a file. There are two arguments in favour of
going from `languages/{language}.py` to instead `languages/{language}/{dataset}.py`: 
1. Autocompletion for the last `.` of the `import` suggests exactly the list of available datasets for that language;
2. You do not have to have all the packages installed required to download/build all the datasets for a language if you 
   only need one. (However, realistically, since MoDeST is for final datasets rather than making datasets, the code for pulling
   them should not be that complicated.)
