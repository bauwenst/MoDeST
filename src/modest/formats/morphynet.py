from typing import Tuple, Iterable
import langcodes

from ..algorithms.alignment import alignMorphemes_Viterbi
from ..interfaces.morphologies import WordDecompositionWithLexicalForm, WordDecomposition


class MorphyNetInflection(WordDecompositionWithLexicalForm):
    """
    Table row looks like:
        désengager	désengageraient	V|COND;3;PL	désengager|eraient
    """

    def __init__(self, word: str, raw_morpheme_sequence: str, lemma: str, lexical_tag: str):
        super().__init__(
            word=word,
            lemma=lemma,
            tag=lexical_tag
        )
        self.morphemes = tuple(raw_morpheme_sequence.split("|"))

    def decompose(self) -> Tuple[str, ...]:
        return self.morphemes


class MorphyNetDerivation(WordDecomposition):
    # TODO: Although we don't save the tag info currently, technically you could
    #       convert MorphyNet's pre- and post-derivation POS tags into CELEX tags.
    #       For example: Wiederverwendbar/keit goes from A to N, so you could turn
    #       it into ( (Wiederverwendbar)[A], (keit)[.A|N] )[N] which lies in between
    #       the tag of the lexical form and having no tag.

    def __init__(self, word: str, base: str, affix: str, prefix_not_suffix: bool):
        """
        MorphyNet's derivational decompositions have a strange loss of information
        where they give the affix and ONE OF the other morphemes, but not always all
        the morphemes, and nothing is ordered. That means you have to figure out
        which part of the word is the affix, which part is the given morpheme, and
        which is the rest, before you can even align them. It gives a flag "prefix"
        or "suffix", yet there are also interfices in the dataset...

        You also can't deduce the full morpheme sequence from the inflectional
        MorphyNet dataset of the same language, because they don't overlap in their
        words. (E.g.: Wiederverwendbarkeit is in German MorphyNet derivations but
        not inflections.)
        """
        super().__init__(
            word=word
        )

        ########################################
        # Old implementation: flawed because it tried to do alignment by itself and couldn't support character edits.
        # try:  # Either it's at the start or at the end.
        #     if prefix_not_suffix:
        #         assert affix.casefold() == surface_form[:len(affix)].casefold()
        #         split_at_index  = len(affix)
        #         replace_by_base = 1
        #     else:
        #         assert affix.casefold() == surface_form[-len(affix):].casefold()
        #         split_at_index  = len(surface_form) - len(affix)
        #         replace_by_base = 0
        #     self.morphs = [surface_form[:split_at_index], surface_form[split_at_index:]]
        # except:  # It might be somewhere in the middle.
        #     replace_by_base = 0
        #     try:   # Immediately after the base.
        #         assert base.casefold() == surface_form[:len(base)].casefold() and affix.casefold() == surface_form[len(base):len(base)+len(affix)].casefold()
        #         split_at_index = len(base)
        #     except:
        #         try:  # Is the affix SOMEWHERE?
        #             split_at_index = surface_form.index(affix)
        #             assert affix.casefold() == surface_form[split_at_index:split_at_index+len(affix)].casefold()
        #         except:
        #             raise RuntimeError(f"Discarded surface form because the affix was weird: '{surface_form}' apparently contains '{affix}'.")
        #     self.morphs = [surface_form[:split_at_index], surface_form[split_at_index:split_at_index+len(affix)], surface_form[split_at_index+len(affix):]]
        # self.morphemes = list(self.morphs)
        # self.morphemes[replace_by_base] = base
        # self.surface = surface_form
        ########################################

        if base[0] == "-":
            base = base[1:]

        if prefix_not_suffix:  # For prefices, we know the prefix comes before the stem. Viterbi figures out which part of each remains.
            morphemes = [affix, base]
            morphs    = alignMorphemes_Viterbi(word, [affix, base])[0].split(" ")
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
            morphemes = [base, affix]
            morphs = alignMorphemes_Viterbi(word, [base, affix])[0].split(" ")
            if len(morphs) > 1 and len(morphs[-1]) > len(affix) and morphs[-1][:len(affix)] == affix:
                last_morph = morphs[-1][len(affix):]
                morphs = morphs[:-1] + [affix, last_morph]
                morphemes.append(last_morph)

        self.morphemes = tuple(morphemes)
        self.morphs    = tuple(morphs)

    def decompose(self) -> Tuple[str, ...]:
        return self.morphemes

    def segment(self) -> Tuple[str, ...]:
        return self.morphs


#######################################################################################


from pathlib import Path

from .tsv import iterateTsv
from ..downloaders.morphynet import MorphynetDownloader, MorphynetSubset
from ..interfaces.datasets import ModestDataset


class MorphyNetDataset_Inflection(ModestDataset):

    def __init__(self, language: langcodes.Language):
        self.language = language

    def _load(self) -> Path:
        dl = MorphynetDownloader()
        return dl.get(language=self.language, subset=MorphynetSubset.INFLECTIONAL)

    def _generate(self, file: Path) -> Iterable[MorphyNetInflection]:
        for parts in iterateTsv(file):
            lemma, word, tag, decomposition = parts
            yield MorphyNetInflection(
                word=word,
                raw_morpheme_sequence=decomposition,
                lemma=lemma,
                lexical_tag=tag
            )


class MorphyNetDataset_Derivation(ModestDataset):

    def __init__(self, language: langcodes.Language):
        self.language = language

    def _load(self) -> Path:
        dl = MorphynetDownloader()
        return dl.get(language=self.language, subset=MorphynetSubset.DERIVATIONAL)

    def _generate(self, path: Path) -> Iterable[MorphyNetDerivation]:
        for parts in iterateTsv(path):
            original, result, original_pos, result_pos, affix, affix_type = parts
            try:
                yield MorphyNetDerivation(
                    word=result,
                    base=original,
                    affix=affix,
                    prefix_not_suffix=(affix_type == "prefix")
                )
            except:
                print("Unparsable MorphyNet derivation:", parts)
                pass
