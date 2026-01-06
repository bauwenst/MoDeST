"""
TODO: You can do a lot of caching at construction. No need to generate the morphemeSplit if you
      know the list of morphemes beforehand.
"""
from dataclasses import dataclass

import re
from tktkt.util.printing import PrintTable, warn

DO_WARNINGS = False

from ..interfaces.morphologies import WordDecompositionWithFreeSegmentation
from ..algorithms.alignment import alignMorphemes_Viterbi


@dataclass
class AlignmentStack:
    current_morpheme: int
    morpheme_indices: list[int]
    morphs: list[str]


class CelexLemmaMorphology(WordDecompositionWithFreeSegmentation):

    POS_TAG = re.compile(r"\[[^\]]+\]")

    def __init__(self, id: int, celex_struclab: str, lemma: str= "", morph_stack: AlignmentStack=None):
        """
        Tree representation of a morphologically decomposed lemma in the CELEX lexicon.
        No guarantees if the input doesn't abide by the specification below.

        :param celex_struclab: morphological tag, e.g. "((kool)[N],(en)[N|N.N],(((centrum)[N],(aal)[A|N.])[A],(e)[N|A.])[N])[N]"
                           These are built hierarchically through the BNF
                                M ::= `(`M,M(,M)*`)`[T]
                           with T something like a PoS tag.
        :param lemma: flat word, e.g. "kolencentrale"
        :param morph_stack: alternative to the lemma, a list containing as its first element the
                            substring in the parent lemma corresponding to this object *if* it has no children.
        """
        if not lemma and not morph_stack:  # This is a very good sanity check that indicates many bugs with the splitter.
            raise ValueError("You can't construct a morphological split without either a lemma or a list of stems for the children to use.")

        self.raw = celex_struclab
        if lemma:
            morphological_split = CelexLemmaMorphology.POS_TAG.sub("", celex_struclab)\
                                                              .replace(" ", "")\
                                                              .replace("(", "")\
                                                              .replace(")", "")\
                                                              .replace(",", " ")\
                                                              .split(" ")
            morph_split,alignment = alignMorphemes_Viterbi(lemma, morphological_split)
            morph_stack = AlignmentStack(current_morpheme=0, morpheme_indices=alignment, morphs=morph_split.split(" "))
            if DO_WARNINGS and len(morph_split.split(" ")) != len(morphological_split):
                warn("Morphemes dropped:", lemma, "--->", celex_struclab, "--->", " ".join(morphological_split), "----->", morph_split)

        raw_body, self.pos, child_strings = CelexLemmaMorphology.parse(celex_struclab)
        self.is_prefix = "|." in self.pos or self.pos == "[P]"
        self.is_suffix = ".]" in self.pos
        self.is_interfix = not self.is_prefix and not self.is_suffix and "." in self.pos
        self.children = [CelexLemmaMorphology(None, sub_celex, morph_stack=morph_stack) for sub_celex in child_strings]
        if self.children:
            self.morphemetext = "+".join([c.morphemetext for c in self.children])
            self.morphtext    =  "".join([c.morphtext    for c in self.children])
            self.retagInterfices()
        else:
            self.morphemetext = raw_body
            self.morphtext = ""  # The concatenation of all morph texts should be the top lemma, so in doubt, set empty.

            has_unaligned_morph = morph_stack.morpheme_indices[0] is None
            is_relevant         = morph_stack.current_morpheme in morph_stack.morpheme_indices
            is_degenerate       = len(morph_stack.morphs) == 1 and has_unaligned_morph  # # VERY rare case where none of the morphemes matched the lemma. Arbitrarily assign the string to the first morpheme, in that case. Could be done better perhaps (e.g. based on best character overlap), but this is rare enough to do it this way.
            if is_relevant or is_degenerate:  # If the i'th leaf is mentioned in the alignment map.
                if has_unaligned_morph:                          # Unaligned substring before the first morpheme is concatenated to the first morph.
                    morph_stack.morpheme_indices.pop(0)          # Remove this unalignment signal.
                    self.morphtext += morph_stack.morphs.pop(0)  # Assign morph
                if not is_degenerate:
                    self.morphtext += morph_stack.morphs.pop(0)  # Assign the actual i'th morph.

            morph_stack.current_morpheme += 1

        super().__init__(id=id, word=self.morphtext)

    def __repr__(self):  # This is some juicy recursion right here
        lines = [self.morphtext + self.pos]
        for c in self.children:
            lines.extend(c.__repr__().split("\n"))
        return "\n|\t".join(lines)

    def toForest(self, do_full_morphemes=False, indent=0):
        s = "[" + (self.morphemetext if do_full_morphemes else self.morphtext) + r" (\textsc{" + self.pos[1:-1].lower().replace("|", r"$\leftarrow$") + "})"
        if self.children is not None:
            s += "\n"
            for child in self.children:
                s += "".join(["\t" + line + "\n" for line in child.toForest(do_full_morphemes=do_full_morphemes, indent=indent + 1).split("\n")])
        s += "]"
        return s

    def printAlignments(self, columns: list=None):
        starting_call = columns is None
        if starting_call:
            columns = []

        if self.children:
            for c in self.children:
                c.printAlignments(columns)
        else:
            columns.append((self.morphemetext,self.morphtext))

        if starting_call:
            t = PrintTable()
            rows = list(zip(*columns))  # Transpose the list.
            t.print(*rows[0])
            t.print(*rows[1])

    def isNNC(self):
        return len(self.children) == 2 and self.children[0].pos == "[N]" and self.children[1].pos == "[N]" \
            or len(self.children) == 3 and self.children[0].pos == "[N]" and self.children[1].is_interfix and self.children[2].pos == "[N]"

    def retagInterfices(self):
        """
        Very rarely, an interfix is followed by a suffix. Whereas we normally split an interfix off of anything else no
        matter the splitting method, it effectively behaves like a suffix in such cases and should be merged leftward if this is desired for suffices.

        An example in e-Lex:
            koppotigen	((kop)[N],(poot)[N],(ig)[N|NN.x],(e)[N|NNx.])[N]
        An example in German CELEX:
            gerechtigkeit  (((ge)[A|.N],((recht)[A])[N])[A],(ig)[N|A.x],(keit)[N|Ax.])[N]

        Note that if the suffix following an interfix is part of a different parent (which is never the case in e-Lex),
        that interfix will not be reclassified as suffix.

        Design note: there are two possible implementations to deal with this.
            1. Add an extra base-case condition
                    if not_second_to_last and self.children[i+1].isInterfix() and self.children[i+2].isSuffix()
               and a method for checking interfices.
            2. Precompute in each CelexLemmaMorphology whether an interfix appears left of a suffix, and store
               in its node that it then IS a suffix.
        """
        i = 0
        while i < len(self.children)-1:
            if self.children[i].is_interfix and self.children[i+1].is_suffix:
                self.children[i].is_suffix = True
            i += 1

    @staticmethod
    def parse(s: str):
        children = []
        tag = ""

        stack = 0
        start_of_child = 0
        for i,c in enumerate(s):
            if c == "(":
                stack += 1
                if stack == 2:
                    start_of_child = i
            elif c == ")":
                stack -= 1
                if stack == 1:
                    children.append(s[start_of_child:s.find("]", i)+1])
                if stack == 0:
                    tag = s[i+1:s.find("]", i)+1]
                    break

        body = s[s.find("(")+1:s.rfind(")")]
        return body, tag, children

    #########################
    ### SPLITTING METHODS ###
    #########################
    ### MORPHEMES ###
    def decompose(self) -> tuple[str, ...]:
        """
        Produces a flat split of all morphemes in the annotation. This is very simple,
        but doesn't match the morphs in the lemma:
            "kolencentrale" is split into the morphemes "kool en centrum aal e"
        but should be split into the morphs "kol en centr al e".
        """
        if self.children:
            return sum((c.decompose() for c in self.children), tuple())
        else:
            return (self.morphemetext,)

    ### LEXEMES ###
    def segmentFree(self) -> tuple[str, ...]:
        """
        Not all morphemes have their own lexeme.

        Generally, lexemeless morphemes have a tag [C|A.B], which means "put between a word of PoS A and PoS B to
        make a new word of PoS C". For suffices, B is dropped. For prefices, A is dropped.

        Examples:
            kelder verdieping    ((kelder)[N],(((ver)[V|.A],(diep)[A])[V],(ing)[N|V.])[N])[N]
            keizers kroon	    ((keizer)[N],(s)[N|N.N],(kroon)[N])[N]
            aanwijzings bevoegdheid	((((aan)[P],(wijs)[V])[V],(ing)[N|V.])[N],(s)[N|N.N],((bevoegd)[A],(heid)[N|A.])[N])[N]
            beziens waardigheid	((((be)[V|.V],(zie)[V])[V],(s)[A|V.A],((waarde)[N],(ig)[A|N.])[A])[A],(heid)[N|A.])[N]
            levens verzekerings overeenkomst (((leven)[N],(s)[N|N.N],(((ver)[V|.A],(zeker)[A])[V],(ing)[N|V.])[N])[N],(s)[N|N.N],(((overeen)[B],(kom)[V])[V],(st)[N|V.])[N])[N]
        """
        return tuple(self._lexemeSplit().replace("  ", " ").replace("  ", " ").strip().split(" "))

    def _lexemeSplit(self) -> str:
        """
        Doing a recursive concatenation became too difficult, so here's a different approach: assume the leaves (which
        are all morphemes) are the final splits. If a leaf is an affix, then it requires its relevant *sibling* to merge
        itself all the way up to that level, no matter how deep it goes.

        You call this method at the top level. The children at that level are superior to the ones at all lower levels;
        it doesn't matter if you have a derivation inside a derivation, because the former will be merged immediately by
        the latter and hence you don't even have to check for it.
        It also means you don't have to be afraid of putting a space left of a prefix, because there can never be another
        prefix to its left (because in that case, that prefix would never allow you to be deciding about that).
        """
        if not self.children:
            return self.morphtext

        s = ""
        for i,child in enumerate(self.children):  # For each child, do a recursive call.
            not_first = i > 0
            not_last  = i < len(self.children)-1
            if (not_first and self.children[i-1].is_prefix) or (not_last and self.children[i+1].is_suffix):  # The boolean flags protect against out-of-bounds errors.
                s += child.morphtext  # Recursive base case: collapse the entire child without spaces.
            else:
                # In the alternative case, you (1) recursively split and (2) add spaces around the result.
                # This is not a hard rule, however, because if the current child is a prefix/suffix, then obviously
                # (2) is wrong. In particular: you should not add a space before a suffix or after a prefix.
                s += " "*(not_first and not child.is_suffix) + child._lexemeSplit() + " "*(not_last and not child.is_prefix)
        return s

    ### MORPHS ###
    def segment(self) -> tuple[str, ...]:
        """
        Splits into morphs rather than morphemes. That is: removing spaces from the result will produce the lemma.
        For the lemma "kolencentrale":
            morphemeSplit: kool en centrum aal e
            morphSplit:    kol en centr al e

        There are two ways to do this: greedily and optimally. Both assume the following word formation process from a
        list (not a set) of morphemes:
            1. For each morpheme, keep it or drop it.
            2. For each kept morpheme, truncate any amount of characters FROM THE END.
            3. Starting from an empty string, alternate between generating random characters and concatenating the next morpheme.
        Dropping morphemes happens surprisingly often (e.g. isolementspositie -> isoleer ement s pose eer itie).

        The most simplifying assumption in this approach is how a morpheme can express itself:
            1. It is a contiguous substring of the morpheme,
            2. if it appears, the first letter is guaranteed,
            3. after the first character mismatch in the string, the rest of the morpheme has no impact whatsoever.

        The goal is now to, given a list of morphemes and the concatenation of the morphs (the lemma), segment back into
        morphs. That is: given a string and a list of strings of which any prefix could be in the first string, find a
        new list of strings which all have a prefix that is also a prefix of a string in the other list, in the same
        order, and whose concatenation is the full string.

        There are two ways to implement this.
            - The greedy approach: for each morpheme, find its biggest prefix that also prefixes the current position
              in the string, and move there. If that prefix has length 0, move over a character, and retry. If you can't
              find any non-zero prefix this way, drop the morpheme and retry the process for the next morpheme.
            - The optimal approach: find the sequence of substrings for which each substring has a prefix matching one
              that of one of the morphemes, the matches are in order, and the sum of character overlaps is maximal.

        All this has to tolerate uppercase lemmas with lowercase morphemes, uppercase morphemes, accented lemmas,
        and hyphenated lemmas.
            - ((andromeda)[N],(nevel)[N])[N]                     Andromedanevel
            - ((Cartesiaan)[N],(s)[A|N.])[A]                     Cartesiaans
            - ((elegant)[A],(nce)[N|A.])[N]                      élégance
            - (((centrum)[N],(aal)[A|N.])[A],(Aziatisch)[A])[A]  centraal-Aziatisch

        For hyphenation, if the hyphen isn't surrounded by characters of the same morph, it is split off by this method.
        Note that this means that the output of this method cannot be aligned with the morpheme list, because there is
        no morpheme for the hyphen. See _morphSplit() for the raw, alignable split.
        """
        split, _ = alignMorphemes_Viterbi(self.morphtext, self.decompose())
        return tuple(split.replace("- ", " - ").split(" "))
