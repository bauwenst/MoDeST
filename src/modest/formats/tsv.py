"""
Tools for TSVs, and particularly those that represent word-count pairs.
"""
from typing import TextIO, Tuple, Optional, Callable, Dict
from collections import Counter
from pathlib import Path

import gc
from tqdm.auto import tqdm

from tktkt.util.timing import *
from tktkt.util.printing import *


def iterateHandle(open_file_handle: TextIO, verbose=False):
    """
    Here's how this function works:
        - Python recognises that a 'yield' is used and not a 'return'. Hence, when you call the function, all that is
          returned is a generator object that has stored the arguments to the function and nothing else.
        - When you iterate over the result of the function call, the first iteration will run until the yield and
          return its result. The next iteration, it will continue running past that yield until it encounters it again.
    """
    open_file_handle.seek(0)
    if verbose:
        # Count total lines
        total_lines = 0
        for _ in open_file_handle:
            total_lines += 1
        open_file_handle.seek(0)

        # Now generate each line whilst updating a progress bar.
        for line in tqdm(open_file_handle, total=total_lines, desc=Path(open_file_handle.name).name, smoothing=0.05):
            yield line.rstrip()
    else:
        for line in open_file_handle:
            yield line.rstrip()


def iterateTsv(tsv_path: Path, sep="\t", n_parts: int=0, verbose=False) -> Iterable[Tuple[str, ...]]:
    """
    Iterating over the words file is slightly trickier than you think due to 2 technicalities that are easy to forget:
        - You must strip the newline at the end;
        - You need to specify a sep=" ", because although Python splits on spaces by default, it uses a special
          algorithm to do so (https://stackoverflow.com/a/30271689/9352077) that drops some Unicode.
          Try " 898".split().

    Hence, we abstract it. The result is completely in TEXT form, even if the second part is a number.
    """
    with open(tsv_path, "r", encoding="utf-8") as handle:
        for stripped_line in iterateHandle(handle, verbose=verbose):
            parts = stripped_line.split(sep=sep)
            if len(parts) >= n_parts:  # Enough parts to output something.
                if n_parts == 1:
                    yield (sep.join(parts),)
                else:
                    yield (sep.join(parts[:-n_parts+1]),) + tuple(parts[-n_parts+1:])

#################################################################################################################

def textIterableToTsv(line_iterable: Iterable[str], output_file: Path,
                      cache_every: int=1_000_000, progress_bar_total: int=None):
    """
    Compresses the given string iterable to an output file, with the result
    containing every unique word exactly once in the format
        word1 count1
        word2 count2
        word3 count3
        ...

    Simplified from get_vocab() at https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_bpe.py.
    """
    from ..paths import PathManagement  # Importing here so that you don't have to deal with path creation for the simple functions in this file.
    CACHE_FOLDER = PathManagement.makeTempFolder()
    CACHE_FOLDER.mkdir(exist_ok=False)

    total_counter = Counter()
    caches = []
    for idx,line in tqdm(enumerate(line_iterable), total=progress_bar_total, smoothing=0.05):
        # Counting
        for word in line.split():  # No strip needed: note that .strip() without arguments will delete ALL WHITESPACE (i.e. any sequence length of space, tab, newline, carriage...). Those newlines would break word files.
            total_counter[word] += 1

        # Caching
        if (idx+1) % cache_every == 0:
            cache_path = CACHE_FOLDER / f"{len(caches)+1}.txt"
            counterToTsv(total_counter, cache_path)
            caches.append(cache_path)
            total_counter = Counter()

    # For safety, cache the current incomplete counter
    if total_counter:
        cache_path = CACHE_FOLDER / f"{len(caches) + 1}.txt"
        counterToTsv(total_counter, cache_path)
        caches.append(cache_path)

    # Merge and delete caches
    mergeTsvs(caches, output_file, delete_afterwards=True, trim_hapax_every=5)
    CACHE_FOLDER.rmdir()

    return output_file


def counterToTsv(counts: Counter, out_path: Path, sep="\t"):
    with open(out_path, "w", encoding="utf-8") as handle:
        for word, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            handle.write(word + sep + str(count) + "\n")
    return out_path


def tsvToCounter(tsv_path: Path, sep="\t") -> Counter:
    c = Counter()
    for word, count in iterateTsv(tsv_path, sep=sep):
        if word not in c:  # Necessary (but not sufficient) to fix a bug where whitespace was left inside words. This condition catches that newlines were added to words, causing something like "a\nb 69" to show up as an ignored line "a" and a line "b 69" which would overwrite any count for b earlier in the file.
            c[word] = int(count)
    return c


@timeit
def mergeTsvs(word_files: List[Path], out_file: Path, delete_afterwards: bool=False,
              trim_hapax_every: int=100000):
    """
    :param trim_hapax_every: To mitigate against very large tails, trim words of count 1 (hapax legomena)
                             every this many files.
    TODO: You could implement an extra "safety valve" that detects when the dictionary goes over a certain size and
          then starts trimming off the tail (count = 1, then 2, then 3, ...) until the size is under the threshold again.
    """
    # Collect
    total_counter = Counter()
    for idx, word_file in enumerate(word_files):
        wprint(f"\nReading word file {word_file.name}...")
        new_counts = tsvToCounter(word_file)

        wprint("Adding counts...")
        for word, count in tqdm(new_counts.items(), total=len(new_counts)):
            total_counter[word] += count

        wprint("Size of total counter:", intsep(len(total_counter)))
        if (idx+1) % trim_hapax_every == 0:
            print("\tTrimming...")
            for word in list(total_counter.keys()):  # A list of all keys is better than a copy of the dictionary.
                if total_counter[word] == 1:
                    del total_counter[word]  # I've read that .pop() doesn't allow garbage collection the same way.
            gc.collect()  # We can't rely on the interpreter to decide when to garbage-collect those del'd items.
            print("\tAfter trimming hapax legomena:", intsep(len(total_counter)))

    # Save
    counterToTsv(total_counter, out_file)

    # Delete
    if delete_afterwards:
        for word_file in word_files:
            word_file.unlink()


@timeit
def trimWordFile(tsv_path: Path, minimum: int) -> Path:
    """
    Removes all words with count < minimum.
    For OSCAR, setting minimum = 10 eliminates about 80% of all words to iterate over, greatly speeding up BPE.
    """
    removed = 0
    new_path = tsv_path.with_stem(tsv_path.stem + "_trimmed")
    with open(new_path, "w", encoding="utf-8") as out_handle:
        for w,c in iterateTsv(tsv_path):
            if int(c) >= minimum:
                out_handle.write(f"{w} {c}\n")
            else:
                removed += 1

    print("Removed", removed, "words from the count file.")
    return new_path


def getSubsetOfAllCounts(tsv_path: Path, subset: Iterable[str], subset_name: str) -> Optional[Counter]:
    """
    Get the intersection between a TSV of counts and a given set of strings.
    Also caches the result.
    """
    if tsv_path is None:  # Impossible to identify which cache file it would be.
        return None

    from ..paths import PathManagement
    cache_path = PathManagement.namelessCache() / f"{tsv_path.stem} ⊗ {subset_name}.txt"  # Path depends on the two files it intersects, otherwise it would be used even if you switched languages.
    if not cache_path.exists():
        if not tsv_path.exists():  # Impossible to fill the cache.
            return None

        counter = Counter()

        # Collect subset
        for w in subset:
            counter[w] = 1  # We effectively add the lexicon to the corpus.

        # Look up their counts
        for word, count in iterateTsv(tsv_path):
            if word in counter:
                counter[word] += int(count)

        # Cache these filtered counts
        with open(cache_path, "w", encoding="utf-8") as handle:
            for word, count in counter.items():
                handle.write(f"{word} {count}\n")

    return tsvToCounter(cache_path)


def loadAndWeight(tsv_path: Path,
                  subset: Iterable[str], subset_name: str,
                  reweighting_function: Callable[[float],float]) -> Dict[str, float]:
    """
    Takes care of converting word counts (integers) to weights (floats)
    and returns a queriable object even if no counts exist.
    """
    lemma_weights = getSubsetOfAllCounts(tsv_path, subset, subset_name)  # Fill the cache.
    if lemma_weights is None:  # Possible if there was no weights file found.
        return dict()
    else:
        lemma_weights = dict(lemma_weights)
        for word, frequency in lemma_weights.items():
            lemma_weights[word] = reweighting_function(frequency)  # Note that it's only disallowed to ADD items in an iterable, not change them.
        return lemma_weights
