import EditDistance
import LanguageModel
import argparse
import pickle
import spacy

class SpellChecker():

    def __init__(self, max_distance, channel_model=None, language_model=None):
        ''' takes in EditDistanceFinder object as channel_model, 
        LanguageModel object as language_model, and an int as max_distance
        to initialize the SpellChecker. '''
        self.nlp = spacy.load("en", pipeline=["tagger", "parser"])
        self.channel_model = channel_model
        self.language_model = language_model
        self.max_distance = max_distance

    def load_channel_model(self, fp):
        ''' Takes in a file pointer as input
        and should initialize the SpellChecker object’s 
        channel_model data member to a default EditDistanceFinder 
        and then load the stored language model (e.g. ed.pkl) 
        from fp into that data member. '''
        self.channel_model = EditDistance.EditDistanceFinder()
        self.channel_model.load(fp)

    def load_language_model(self, fp):
        ''' Takes in a file pointer as input and should initialize the 
        SpellChecker object’s language_model data member to a default 
        LanguageModel and then load the stored language model (e.g. lm.pkl) 
        from fp into that data member. ''' 
        self.language_model = LanguageModel.LanguageModel()
        self.language_model.load(fp)

    def bigram_score(self, prev_word, focus_word, next_word):
        ''' Take in 3 words and return average of bigram probability 
        for both bigrams of the sequence. '''
        score1 = self.language_model.bigram_prob(prev_word, focus_word)
        score2 = self.language_model.bigram_prob(focus_word, next_word)
        return (score1 + score2)/2

    def unigram_score(self, word):
        ''' Take a word as input and return the unigram probability of 
        the word according to the LanguageModel '''
        return self.language_model.unigram_prob(word)
        
    def inserts(self, word):
        ''' Take a word as input and return a list of words (that are in 
        the LanguageModel) that are within one insert of word.'''
        within_one_insert = []
        for intended_word in self.language_model.vocabulary:
            # if we inserted 1 char to get from observed word to intended_word
            if len(word) + 1 == len(intended_word):
                alignment = self.channel_model.align(word, intended_word)
                distance, tuples = self.channel_model.align("suprise", "surprise")
                observed = ""
                intended = ""
                perc_count = 0
                for t in tuples:
                    if perc_count > 1: break
                    if t[0] != t[1] or t[0] != "%": break
                    if t[0]=="%":
                        perc_count += 1
                    observed += t[0]
                    intended += t[1]
                print(observed)
                print(intended)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ed", type=argparse.FileType('rb'))
    parser.add_argument("--lm", type=argparse.FileType('rb'))
    args = parser.parse_args()

    sp = SpellChecker(max_distance = 2)
    sp.load_channel_model(args.ed)