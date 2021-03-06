import EditDistance
from LanguageModel import LanguageModel
import argparse
import pickle
import spacy
import string
from spacy.lang.en import English
import string

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
        self.language_model = LanguageModel()
        self.language_model.load(fp)

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
            if all(c in string.ascii_lowercase for c in posWord):
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


    def generate_candidates(self, word):
        
        ''' Takes a word as input and returns a list of
        words that are within self.max_distance edits of word 
        by calling inserts, deletes, and substitutions '''
        potentials = []
        potentials.extend(self.inserts(word))
        potentials.extend(self.deletes(word))
        potentials.extend(self.substitutions(word))
        n = 1

        temp = []

        while n < self.max_distance:
            for pot in potentials:
                temp.extend(self.inserts(pot))               # as we go, dedupe temp!
                temp.extend(list(filter(lambda x: x not in temp, self.deletes(pot))))
                temp.extend(list(filter(lambda x: x not in temp, self.substitutions(pot))))
            # make sure none of the things in temp are already in potentials!
            potentials.extend(list(filter(lambda x: x not in potentials, temp)))
            temp = []
            n += 1

        # transposition!
        for index in range(0, len(word)-1):
            swap = word[0:index] + word[index+1] + word[index] + word[index+2:]
            if swap in self.language_model.vocabulary:
                potentials.append(swap)
        return potentials


    def unigram_score(self, word):
        ''' Take a word as input and return the unigram probability of 
        the word according to the LanguageModel '''
        return self.language_model.unigram_prob(word)
        
    def inserts(self, word):
        ''' Take a word as input and return a list of words (that are in 
        the LanguageModel) that are within one insert of word.'''

        within_one_insert = []

        for intended_word in self.language_model.vocabulary:

            if all(c in string.ascii_lowercase for c in intended_word):
            # only consider intended words whose length is exactly 1 greater than word
                
                if len(word) + 1 == len(intended_word):

                    distance, tuples = self.channel_model.align(word, intended_word)
                    insert_count = 0
                    doesnt_work = False

                    for t in tuples:
                        if insert_count > 1:                # needing to insert more than once
                            doesnt_work = True              # means this intended word doesn't work
                            break
                        if t[0] != t[1] and t[0] != "%":    # if we needed a substitution or deletion
                            doesnt_work = True              # then this intended word doesn't work
                            break
                        if t[0]=="%":                       # we're only allowed one insertion, so
                            insert_count += 1               # need to keep track

                    if not doesnt_work:                          # as long as intended word works,
                        within_one_insert.append(intended_word)  # add it to list

        return within_one_insert

    def check_sentence(self, sentence, fallback=False):
        ''' Takes in list of words and returns a list of lists
        such that each sublist in the returned lists corresponds
        to a single word in the input. For each word in input, 
        if it's in the language model, its sublist in the output
        will just contain that word. Otherwise, its sublist will
        be a list of possible corrections sorted from most likely
        to least likely (combo of LangModel and EditDist scores) '''
        suggestions = [] 
        sentence.insert(0, '<s>')
        sentence.append('</s>')
        for index in range(1, len(sentence)-1):
            word = sentence[index]
            if word in self.language_model.vocabulary:
                suggestions.append([word])
            else:
                corrections = self.generate_candidates(word)
                weighted = []
                for item in corrections:
                    edprob = self.channel_model.prob(word, item)
                    lmprob1 = self.unigram_score(item)
                    lmprob2 = self.bigram_score(sentence[index-1], item, sentence[index+1])
                    avg = (lmprob1+lmprob2)/2.0
                    weighted.append(((edprob+avg), item))
                sortedCorrections = sorted(weighted, key=lambda x: x[0], reverse=True)
                corrections = [item[1] for item in sortedCorrections]
                if len(corrections) == 0 and fallback:
                    suggestions.append([word])
                else:
                    suggestions.append(corrections)
        return suggestions

    def check_text(self, text, fallback=False):
        ''' Takes in string as input, tokenizes and sentence
        segments it with spacy, then returns the concatenated
        result of calling check_sentence on all of the resulting
        sentence objects ''' 
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        doc = nlp(text)
        sents = list(doc.sents)
        text = []
        for sent in sents:
            wordList = [t.text for t in sent]
            checked = self.check_sentence(wordList, fallback)
            text.extend(checked)
        return text

    def autocorrect_sentence(self, sentence):
        ''' Takes in a tokenized sentence as a list of
        words, calls check_sentence on that sentence, and
        returns list of tokens where each non-word has been
        replaced by its most likely spelling correction '''
        suggestions = self.check_sentence(sentence, fallback=True)
        return [sublist[0] for sublist in suggestions]

    def autocorrect_line(self, line):
        ''' Takes in string as input, tokenizes and sentence
        segments it with spacy, then returns the concatenated
        result of calling autocorrect_sentence on all of the 
        resulting sentence objects ''' 
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        doc = nlp(line)
        sents = list(doc.sents)
        punc = [s[-1] for s in sents]       # save end of sentence punctuation
        sents = [s[:-1] for s in sents]     # get rid of end of sentence punctuation 
        text = []
        for i in range(len(sents)):
            if len(sents[i]) > 0:
                wordList = [t.text for t in sents[i]]
                wordList = [w.lower() for w in wordList]               # get rid of capitalization
                wordList = [''.join(ch for ch in word if ch not in set(string.punctuation)) for word in wordList]
                wordList = list(filter(lambda x: x != "", wordList))   # get rid of things that only consisted of punc
                checked = self.autocorrect_sentence(wordList)
                checked[-1] += str(punc[i])                            # replace punctuation at end
                checked[0] = checked[0][0].upper() + checked[0][1:]    # capitalize first character 
                text.extend(checked)
        return text

    def suggest_sentence(self, sentence, max_suggestions):
        ''' Takes in list of words as input, calls check_sentence on it,
        and returns a list where real words are just strings in the list
        and non-words are represented by lists of corrections limited to
        max_suggestions number of suggestions. '''
        corrections = self.check_sentence(sentence)
        limited = []
        for correct in corrections:
            if len(correct) == 1:
                limited.append(correct[0])
            else:
                limited.append(correct[:max_suggestions])

        return limited


    def suggest_text(self, text, max_suggestions):
        ''' Takes in a string as input, tokenizes and segments it with
        spacy, then returns the concatenation of the result of calling
        suggest_sentence on all of the resulting sentence objects '''
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        doc = nlp(text)
        sents = list(doc.sents)
        text = []
        for sent in sents:
            wordList = [t.text for t in sent]
            checked = self.suggest_sentence(wordList, max_suggestions)
            text.extend(checked)

        return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ed", type=argparse.FileType('rb'))
    parser.add_argument("--lm", type=argparse.FileType('rb'))
    args = parser.parse_args()

    sp = SpellChecker(max_distance=1)
    sp.load_channel_model(args.ed)
    sp.load_language_model(args.lm)



 
