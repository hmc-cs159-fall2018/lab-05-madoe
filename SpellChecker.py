import EditDistance
import LanguageModel
import pickle

class SpellChecker():

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