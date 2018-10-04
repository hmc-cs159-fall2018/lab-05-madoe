import EditDistance
import LanguageModel
import pickle

class SpellChecker():

    def __init__(self, channel_model=None, language_model=None, max_distance):
        ''' takes in EditDistanceFinder object as channel_model, 
        LanguageModel object as language_model, and an int as max_distance
        to initialize the SpellChecker. '''
        self.nlp = spacy.load("en", pipeline=["tagger", "parser"])
        self.channel_model = channel_model
        self.language_model = language_model
        self.max_distance = max_distance

    def load_channel_model(fp):
        ''' Takes in a file pointer as input
        and should initialize the SpellChecker object’s 
        channel_model data member to a default EditDistanceFinder 
        and then load the stored language model (e.g. ed.pkl) 
        from fp into that data member. '''
        self.channel_model = EditDistance.EditDistanceFinder()
        self.channel_model = pickle.load(fp)

    def load_language_model(fp):
        ''' Takes in a file pointer as input and should initialize the 
        SpellChecker object’s language_model data member to a default 
        LanguageModel and then load the stored language model (e.g. lm.pkl) 
        from fp into that data member. ''' 
        self.language_model = LanguageModel.LanguageModel()
        self.language_model = pickle.load(fp)

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
        