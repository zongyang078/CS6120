# Q1 - read_vocabulary
import re
from collections import Counter
from typing import List

def read_vocabulary(filename: str) -> List[str]:
    """
    Reads in a given file specified by "filename" and processes it
    by removing punctuation, forcing lowercase, splits into
    individual words, and removes the numbers that might appear in
    the text.

    Args:
        filename: the name of the file to be processed

    Returns:
        A list of UNIQUE words ordered by frequency (most -> least).
        Example: ['the', 'and', 'i', ...]
    """
    # 1) Read file as UTF-8
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Lowercase
    text = text.lower()

    # 3) Remove digits (numbers anywhere in the text)
    text = re.sub(r"\d+", " ", text)

    # 4) Remove punctuation:
    # Keep only letters and whitespace. Replace everything else with space.
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5) Split by whitespace to get words
    words = text.split()

    # 6) Count frequencies
    freq = Counter(words)

    # 7) Sort unique words by frequency (descending), then lexicographically for stable ties
    sorted_words = sorted(freq.keys(), key=lambda w: (-freq[w], w))

    return sorted_words


# Q2 - autocomplete_word
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    top: List[str] = field(default_factory=list)  # top-10 words for this prefix

def process_data(word_list: List[str]) -> Tuple[TrieNode, Dict[str, int]]:
    """
    Builds a fast autocomplete data structure from a list of UNIQUE words ordered
    by frequency (most frequent -> least).

    Args:
        word_list: A list of unique words sorted by decreasing frequency.

    Returns:
        model_or_data_structure: (root_trie_node, rank_dict)
        - rank_dict[word] = its position in word_list (smaller = more frequent)
    """
    root = TrieNode()
    rank = {w: i for i, w in enumerate(word_list)}  # frequency rank proxy

    def add_to_top(node: TrieNode, w: str) -> None:
        """Maintain node.top as top-10 by frequency rank (smaller index = higher freq)."""
        if w in node.top:
            return
        node.top.append(w)
        node.top.sort(key=lambda x: rank[x])
        if len(node.top) > 10:
            node.top.pop()

    # Insert each word into trie; update top-10 cache along the path
    for w in word_list:
        if not w:
            continue
        node = root
        add_to_top(node, w)
        for ch in w:
            if not ('a' <= ch <= 'z'):
                continue
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            add_to_top(node, w)

    return root, rank

def autocomplete_word(prefix: str, model_or_data_structure) -> List[str]:
    """
    Returns a list of words starting with the given prefix.
    This list is sorted in order of the frequency (probability) of the words.

    Args:
        prefix: The prefix to search for.
        model_or_data_structure: output of process_data(word_list).

    Returns:
        A list of ten most-common words starting with the prefix.
    """
    root, _rank = model_or_data_structure

    # Normalize prefix similarly to Q1 expectations: lowercase + keep letters only
    if prefix is None:
        prefix = ""
    prefix = prefix.lower()
    prefix = "".join(ch for ch in prefix if 'a' <= ch <= 'z')

    node = root
    if prefix == "":
        return list(node.top)

    for ch in prefix:
        if ch not in node.children:
            return []
        node = node.children[ch]

    return list(node.top)


# test
def run_basic_tests():
    vocab = read_vocabulary("shakespeare-edit.txt")
    model = process_data(vocab)

    tests = [
        ("th", "common prefix"),
        ("the", "full word prefix"),
        ("love", "meaningful word prefix"),
        ("", "empty prefix (top10 overall)"),
        ("qz", "non-existent prefix"),
        ("Th", "case-insensitive input"),
        ("th!", "punctuation in input"),
        ("th2", "digits in input"),
    ]

    for prefix, note in tests:
        result = autocomplete_word(prefix, model)
        print(f"[{note}] prefix={prefix!r}")
        print(" ->", result)
        print(f" -> count={len(result)} (should be <= 10)")
        print("-" * 60)

if __name__ == "__main__":
    run_basic_tests()

# Sample test output：
# [common prefix] prefix='th'
#  -> ['the', 'that', 'this', 'thou', 'thy', 'thee', 'they', 'then', 'there', 'their']
#  -> count=10 (should be <= 10)
# ------------------------------------------------------------
# [full word prefix] prefix='the'
#  -> ['the', 'thee', 'they', 'then', 'there', 'their', 'them', 'these', 'therefore', 'themselves']
#  -> count=10 (should be <= 10)
# ------------------------------------------------------------
# [meaningful word prefix] prefix='love'
#  -> ['love', 'loves', 'lover', 'lovers', 'lovely', 'loved', 'lovell', 'lovel', 'lovest', 'loveth']
#  -> count=10 (should be <= 10)
# ------------------------------------------------------------
# [empty prefix (top10 overall)] prefix=''
#  -> ['the', 'and', 'i', 'to', 'of', 'a', 'you', 'my', 'that', 'in']
#  -> count=10 (should be <= 10)
# ------------------------------------------------------------
# [non-existent prefix] prefix='qz'
#  -> []
#  -> count=0 (should be <= 10)
# ------------------------------------------------------------
# [case-insensitive input] prefix='Th'
#  -> ['the', 'that', 'this', 'thou', 'thy', 'thee', 'they', 'then', 'there', 'their']
#  -> count=10 (should be <= 10)
# ------------------------------------------------------------
# [punctuation in input] prefix='th!'
#  -> ['the', 'that', 'this', 'thou', 'thy', 'thee', 'they', 'then', 'there', 'their']
#  -> count=10 (should be <= 10)
# ------------------------------------------------------------
# [digits in input] prefix='th2'
#  -> ['the', 'that', 'this', 'thou', 'thy', 'thee', 'they', 'then', 'there', 'their']
#  -> count=10 (should be <= 10)
# ------------------------------------------------------------