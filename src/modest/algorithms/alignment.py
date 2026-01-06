from typing import Iterable
from dataclasses import dataclass

import tokenizers.normalizers as tn
normalizer = tn.Sequence([tn.NFD(), tn.StripAccents(), tn.NFKC()])


@dataclass
class ViterbiNode:
    best_count: int = -1  # Ensures that the backpointer is initialised by the first node that talks to this node.
    backpointer: tuple[int, int] = None


def alignMorphemes_Greedy(word: str, morphemes: Iterable[str]) -> str:
    """
    The greedy approach sometimes outputs the wrong split, namely when a morpheme's tail looks like the next morpheme.
    An example is "élégance -> ((elegant)[A],(nce)[N|A.])[N]", where the "n" in "élégance" supposedly does not come
    from the first morpheme but from the second. The greedy approach first finds "élégan" as the first morpheme's
    match, and then cannot find the morpheme "nce" in the remainder "ce".
    e-Lex contains 2848 such cases. Some other examples:
        acceptatiegraad   ---> accept eer atie graad    ---> acceptati egr aad
        academievriend    ---> academisch ie vriend     ---> academievr iend
        protestantsgezind ---> protest eer ant s gezind ---> protestantsg ezind
    To clarify: in the last example, it finds "protestant". Then it tries to find any prefix of "eer", and indeed,
    it finds an "e", but only later on. That means the letters between protestant and e, "sg", are stuck to the former.
    Then it tries to find the remaining morphemes and finds none of them, so it sticks the rest to that e.

    Can you always detect that greedy has made this mistake by counting morphs? No. There will be both false positives
    but also false negatives.
        False positive: isoleer ement s pose eer itie  has morphemes that disappear in  isole ment s pos itie
        False negative: A BC C D  accidentally subsumes the first C of  A BCE C D  in the BC morpheme, and matches it with the second. This is clearly unintended.
    """
    matching_lemma     = normalizer.normalize_str(word).lower()
    matching_morphemes = map(lambda m: normalizer.normalize_str(m).lower(), morphemes)

    result = ""
    big_cursor = 0
    for part in matching_morphemes:
        # Move until you get to the part; if it is nowhere, try again for the next part.
        big_cursor_cache = big_cursor
        try:
            while part[0] != matching_lemma[big_cursor]:
                big_cursor += 1
        except IndexError:
            big_cursor = big_cursor_cache
            continue
        result += word[big_cursor_cache:big_cursor]

        # Expand into biggest prefix of part
        i = 0
        while part.startswith(matching_lemma[big_cursor:big_cursor + i + 1]) and big_cursor + i < len(word):
            i += 1
        result += " " * (len(result) != 0) + word[big_cursor:big_cursor + i]
        big_cursor += i

    result += word[big_cursor:]
    return result


def alignMorphemes_Viterbi(word: str, morphemes: Iterable[str]) -> tuple[str, list[int]]:  # TODO: Should be a list of strings, not just one string. String concatenation is more expensive than list appends.
    """
    Iterative Viterbi algorithm with the same optimal results as a recursive bruteforce, except the problem goes
    from completely intractable (several minutes, running out of memory) to trivial (0 seconds and a small table).

    Viterbi is NOT as simple as e.g. in the BBPE paper for decoding bad UTF-8. The reason is that the allowed
    vocabulary (the set of steps to new nodes) CHANGES depending on which steps have been made on the path to a
    node, meaning you can't be sure which solution is optimal up to that node without knowing what happens after.

    Here's how I re-interpreted the problem to Viterbi: instead of a substring by itself being a node in
    the search graph, a node is a pair of (substring, available vocab), where 'available vocab' is the start of the
    sublist of morphemes remaining, out of all morphemes available.
    """
    # Normalising does not change the amount of characters in the strings. We normalise to compute the alignment and
    # then, at the end, use substring length to read from the unnormalised string.
    word_normed      = normalizer.normalize_str(word).lower()
    morphemes_normed = map(lambda m: normalizer.normalize_str(m).lower(), morphemes)

    morpheme_prefices = [[morpheme[:i] for i in range(len(morpheme) + 1)]
                         for morpheme in morphemes_normed]
    n_morphemes = len(morpheme_prefices)
    n_chars     = len(word_normed)
    n_rows_trellis = n_morphemes + 1  # You can have used 0, 1, ..., all morphemes.
    n_cols_trellis = n_chars + 1      # Column i shows the best path to get to character i (starting at 0).
                                      # You need an "end-character" column to traverse the whole string.

    trellis = [  # Note that the trellis is indexed with transposed indices a.o.t. a matrix.
        [ViterbiNode() for _ in range(n_rows_trellis)]
        for _ in range(n_cols_trellis)
    ]
    for n_morphemes_expended in range(n_rows_trellis):
        trellis[0][n_morphemes_expended].best_count = 0  # Better than -1, the default for all the following nodes.

    # Forward pass
    for char_idx in range(n_chars):  # The last column isn't solved, but only stored in.
        for n_morphemes_expended in range(n_rows_trellis):
            # You now know which search node you are at. You will
            # now try to offer yourself to all reachable nodes.
            current_node = trellis[char_idx][n_morphemes_expended]

            if n_morphemes_expended < n_morphemes:
                # Reachable set 1: anything an available prefix allows.
                for prefix in morpheme_prefices[n_morphemes_expended]:
                    if word_normed[char_idx:].startswith(prefix):
                        # You offer yourself to the node with 1 more
                        # morphemes expended and one prefix ahead.
                        amount_covered = len(prefix)
                        score_after_step = current_node.best_count + amount_covered

                        new_char_idx = char_idx + amount_covered
                        new_n_morphemes = n_morphemes_expended + 1

                        new_node = trellis[new_char_idx][new_n_morphemes]
                        if new_node.best_count < score_after_step:
                            new_node.best_count = score_after_step
                            new_node.backpointer = (char_idx, n_morphemes_expended)

                # Reachable set 2: skipping any amount of characters.
                for new_char_idx in range(char_idx + 1, n_cols_trellis):
                    # Don't allow dropping a morpheme. The reason is
                    # that a node already attempts to do that itself
                    # by moving vertically in the table.
                    new_node = trellis[new_char_idx][n_morphemes_expended]
                    if new_node.best_count < current_node.best_count:
                        new_node.best_count = current_node.best_count
                        new_node.backpointer = (char_idx, n_morphemes_expended)
            else:  # You can only skip. It is pointless to skip in many steps, so go right to the end.
                new_node = trellis[-1][-1]
                if new_node.best_count < current_node.best_count:
                    new_node.best_count = current_node.best_count
                    new_node.backpointer = (char_idx, n_morphemes_expended)

    # Backward pass
    # - Find best node in the last column by maxing on a double key:
    #   in case of a tie, the one with the most morphemes expended wins.
    col_idx = n_cols_trellis - 1
    row_idx = max(range(n_rows_trellis), key=lambda row: (trellis[col_idx][row].best_count, row))
    node = trellis[col_idx][row_idx]

    # - Build string
    morph_split = ""
    alignment = []

    # trace = [(col_idx,row_idx)]
    while node.backpointer is not None:
        new_col_idx, new_row_idx = node.backpointer

        is_start_of_morpheme = new_row_idx != row_idx and new_col_idx != col_idx  # You consumed a morpheme, and more than 0 characters of it.
        morph_split = " " * (is_start_of_morpheme and new_col_idx != 0) + word[new_col_idx:col_idx] + morph_split  # If you stayed on the same row, the added substring was caused by a skip, not by a recognised prefix.
        if is_start_of_morpheme:
            alignment.append(new_row_idx)
        elif new_col_idx == 0:   # You skip to the start. Special case where the lemma doesn't start with any morpheme.
            alignment.append(None)  # Arguably this should be aligned to the first morpheme that has been aligned with.

        col_idx, row_idx = new_col_idx, new_row_idx
        node = trellis[col_idx][row_idx]
        # trace.append((col_idx,row_idx))
    # viterbiLaTeX(trellis, lemma, morphemes, trace)

    alignment.reverse()
    return morph_split, alignment
