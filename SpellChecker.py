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

    def load_channel_model(self, fp):
        ''' Takes in a file pointer as input
        and should initialize the SpellChecker object’s 
        channel_model data member to a default EditDistanceFinder 
        and then load the stored language model (e.g. ed.pkl) 
        from fp into that data member. '''
        self.channel_model = EditDistance.EditDistanceFinder()
        self.channel_model = pickle.load(fp)

    def load_language_model(self, fp):
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

    def cm_score(self, error_word, corrected_word):
        ''' Takes in a misspelled word and uses the EditDistanceFinder object
        we have as channel_model to get the probability that corrected_word
        was the intended word. '''
        prob = self.channel_model.prob(error_word, corrected_word)
        return prob


    def deletes(self, word):
        ''' Takes in a potentially misspelled word and checks for 
        words in the language model vocabulary that are within one 
        delete of word, i.e. 'hair' is within one delete of 'hairr'. '''
        potentialWords = []
        for index in range(0, len(word)):
            spliced = word[0:index] + word[index+1:]
            if self.language_model.__contains__(spliced):
                potentialWords.append(spliced)

        return potentialWords


    def substitutions(self, word):
        ''' Take in potentially misspelled word and find list of 
        words from the language_model vocabulary which are within 
        one substitution of word, i.e. 'heve' is within one substitution
        of 'have'. '''
        potentialWords = []
        length = len(word)
        for posWord in self.language_model.vocabulary:
            #we only want to examine words of the same length
            if len(posWord) == length:
                #counter for differences between words
                diffs = 0
                #loop through all the characters
                for char1, char2 in zip(word, posWord):
                    #if the characters are different
                    if char1 != char2:
                        diffs += 1
                    #if more than one difference, its not 1 substitution away!
                    if diffs > 1:
                        break
                #if within 1 substitution
                if diffs < 2:
                    potentialWords.append(posWord)

        return potentialWords 


    def generate_candidates(word):

        potentials = []
        potentials.extend(self.inserts(word))
        potentials.extend(self.deletes(word))
        potentials.extend(self.substitutions(word))
        n = 1

        temp = []

        while n < self.max_distance:
            for pot in potentials:
                temp.extend(self.inserts(pot))
                temp.extend(self.deletes(pot))
                temp.extend(self.substitutions(pot))
            potentials = temp
            temp = []
            n += 1

        return potentials
