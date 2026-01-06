from ..algorithms.alignment import alignMorphemes_Viterbi
from ..interfaces.morphologies import WordDecompositionWithLexicalForm, WordDecomposition


class MorphyNetInflection(WordDecompositionWithLexicalForm):
    """
    Table row looks like:
        désengager	désengageraient	V|COND;3;PL	désengager|eraient
    """

    def __init__(self, id: int, word: str, raw_morpheme_sequence: str, lemma: str, lexical_tag: str):
        super().__init__(
            id=id,
            word=word,
            lemma=lemma,
            tag=lexical_tag
        )
        self.morphemes = tuple(raw_morpheme_sequence.split("|"))

    def decompose(self) -> tuple[str, ...]:
        return self.morphemes


class MorphyNetDerivation(WordDecomposition):
    # TODO: Although we don't save the tag info currently, technically you could
    #       convert MorphyNet's pre- and post-derivation POS tags into CELEX tags.
    #       For example: Wiederverwendbar/keit goes from A to N, so you could turn
    #       it into ( (Wiederverwendbar)[A], (keit)[.A|N] )[N] which lies in between
    #       the tag of the lexical form and having no tag.

    def __init__(self, id: int, word: str, base: str, affix: str, prefix_not_suffix: bool):
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
        super().__init__(id=id, word=word)

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

        self._morphemes = tuple(morphemes)
        self._morphs    = tuple(morphs)

    def decompose(self) -> tuple[str, ...]:
        return self._morphemes

    def segment(self) -> tuple[str, ...]:
        return self._morphs
