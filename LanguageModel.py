##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

# coding: utf-8

import re
from math import log
import spacy
from collections import Counter, defaultdict
import pickle
import argparse

nlp_full = spacy.load('en', pipeline=['tagger','parser'])
nlp_basic = spacy.load('en', pipeline=[])
wordRE = re.compile(r'\w')

class LanguageModel():
    def __init__(self, alpha=0.1, max_vocab = 40000):
        self.vocabulary=set()
        self.V = 0
        self.alpha = alpha
        self.max_vocab = max_vocab

        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.vocabulary = set()

    def save(self, fp):
        pickle.dump(self, fp)

    def load(self, fp):
        other = pickle.load(fp)
        self.unigrams = other.unigrams
        self.bigrams = other.bigrams
        self.vocabulary = other.vocabulary

        self.V = other.V
        self.alpha = other.alpha
        self.max_vocab = other.max_vocab

    def set_vocab(self, source_files):
        print("Set vocab...")
        vocab = Counter()

        for chunk in self.get_chunks(source_files):
            vocab.update(self.get_tokens(nlp_basic(chunk)))
            
        self.vocabulary = set([x[0] for x in vocab.most_common(self.max_vocab)] + ["<s>", "</s>", "UNK"])
        self.V = len(self.vocabulary)

    def get_chunks(self, source_files, chunk_size=100000):
        print("Get Chunks...")
        for fp in source_files:
            print("Reading from %s" % (fp.name))
            fp.seek(0)
            chunk = True
            while chunk:
                chunk = fp.readlines(chunk_size)
                if chunk: yield "\n".join(chunk)
        
    def get_tokens(self, sentence):
        return [x.text.lower() for x in sentence if wordRE.search(x.text)]

    def train(self, source_files):
        self.set_vocab(source_files)
        self.set_probs(source_files)

    def set_probs(self, source_files):
        for chunk in self.get_chunks(source_files): 
            doc = nlp_full(chunk)

            for sentence in doc.sents:
                words=["<s>",] + self.get_tokens(sentence) + ["</s>",]
                words = [x if x in self.vocabulary else "UNK" for x in words]
            
                self.unigrams.update(words)
                for w1, w2 in zip(words[:-1], words[1:]):
                    self.bigrams[w1].update([w2])
        
    def bigram_prob(self, w1, w2):
        first_word = w1 if w1 in self else "UNK"
        second_word = w2 if w2 in self else "UNK"
        numerator = self.bigrams[first_word][second_word] + self.alpha
        denominator = sum(self.bigrams[first_word].values()) + (self.alpha * self.V)
        return log(numerator / denominator)

    def unigram_prob(self, w):
        word = w if w in self else "UNK"
        numerator = self.unigrams[word] + self.alpha
        denominator = sum(self.unigrams.values()) + (self.alpha * self.V)
        return log(numerator/denominator)
    
    def __contains__(self, w):
        return w in self.vocabulary

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--store", "-s", type=argparse.FileType('wb'), required=True)
    parser.add_argument("--alpha", "-a", type=float, default="0.1")
    parser.add_argument("--vocab", "-v", type=int, default="40000")
    parser.add_argument("source", type=argparse.FileType('r', encoding='UTF-8'), nargs="+")

    args = parser.parse_args()

    lm = LanguageModel(alpha=args.alpha, max_vocab = args.vocab)

    lm.train(args.source)

    lm.save(args.store)
