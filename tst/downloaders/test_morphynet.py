import langcodes

from modest.algorithms.alignment import alignMorphemes_Viterbi
from modest.datasets.morphynet import MorphynetDownloader, MorphynetSubset
from modest.formats.morphynet import MorphyNetDerivation
from modest.formats.tsv import iterateTsv


def get_examples():
    #
    dl = MorphynetDownloader()
    english = dl.get(language=langcodes.find("English"), subset=MorphynetSubset.INFLECTIONAL)
    french = dl.get(language=langcodes.find("French"), subset=MorphynetSubset.INFLECTIONAL)
    try:
        dl.get(language=langcodes.find("Dutch"), subset=MorphynetSubset.INFLECTIONAL)
    except:
        print("No Dutch in MorphyNet")
    german = dl.get(language=langcodes.find("German"), subset=MorphynetSubset.DERIVATIONAL)
    italian = dl.get(language=langcodes.find("Italian"), subset=MorphynetSubset.DERIVATIONAL)
    spanish = dl.get(language=langcodes.find("Spanish"), subset=MorphynetSubset.DERIVATIONAL)

    #
    for thing in iterateTsv(english):
        print(thing)
        break
    for thing in iterateTsv(french):
        print(thing)
        break
    for thing in iterateTsv(german):
        print(thing)
        break
    for thing in iterateTsv(italian):
        print(thing)
        break

    for thing in MorphyNetDerivation.generator(german):
    # for thing in MorphyNetDerivation.generator(italian):
    # for thing in MorphyNetDerivation.generator(spanish):
        print(thing.lemma(), '->', thing.morphemes, '->', thing.morphSplit())


def drafting():
    from modest.formats.tsv import iterateTsv

    dl = MorphynetDownloader()
    path = dl.get(language=langcodes.find("German"), subset=MorphynetSubset.DERIVATIONAL)
    for parts in iterateTsv(path):
        affix = parts[4]
        word = parts[1]
        stem = parts[0]

        if stem.startswith("-"):
            stem = stem[1:]

        if affix in {"ier", "s"}:
            print(parts)
            if parts[5] == "prefix":  # For prefices, we know the prefix comes before the stem. Viterbi figures out which part of each remains.
                print("\t", alignMorphemes_Viterbi(word, [affix, stem])[0])
            else:  # For "suffices", we actually don't know if there are two or three pieces.
                # There is a tricky problem with interfices. You know the morpheme sequence looks like
                # [stem, affix, something] but that 'something' can only be determined if you already
                # know the alignment of the stem and the affix so that you know where the affix is in
                # the word to then select the rest of the word as a stand-in for the final morpheme.
                #
                # As an example: ('Pose', 'posieren', 'N', 'V', 'ier', 'suffix')
                #
                # The way I solve this is by simply applying Viterbi using only the stem and affix as morphemes.
                # The last morph this will produce contains both the given suffix and possibly more characters after
                # those, which we can split off as an extra morph.
                segmentation, _ = alignMorphemes_Viterbi(word, [stem, affix])
                morphs = segmentation.split(" ")
                if len(morphs) > 1 and len(morphs[-1]) > len(affix) and morphs[-1][:len(affix)] == affix:
                    print("\t!", morphs[:-1] + [affix, morphs[-1][len(affix):]])
                else:
                    print("\t", morphs)


if __name__ == "__main__":
    get_examples()