##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##


from SpellCheck import SpellChecker
from LanguageModel import LanguageModel
from EditDistance import EditDistanceFinder
import argparse
import pickle
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languagemodel", "-l", type=argparse.FileType('rb'), required=True)
    parser.add_argument("--editmodel", "-e", type=argparse.FileType('rb'), required=True)
    args = parser.parse_args()

    s=SpellChecker(max_distance=2)
    s.load_language_model(args.languagemodel)
    s.load_channel_model(args.editmodel)

    print(s.channel_model.prob("hello", "hello"))
    print(s.channel_model.prob("hellp", "hello"))
    print(s.channel_model.prob("hllp", "hello"))

    print(s.check_text("they did not yb any menas"))
    """
    >>> [['they'], ['did'], ['not'], ['by', 'b', 'ye', 'y', 'yo', 'ob', 'ya', 'ab'], ['any'], 
    >>>  ['means', 'mens', 'mena', 'zenas', 'menan', 'mends']]
    """

    print(s.autocorrect_line("they did not yb any menas"))
    """
    >>> ['they', 'did', 'not', 'by', 'any', 'means']
    """

    print(s.suggest_line("they did not yb any menas", max_suggestions=2))
    """
    >>> ['they', 'did', 'not', ['by', 'b'], 'any', ['means', 'mens']]
    """
