1. In Writeup.md, explain how Laplace smoothing works in general and how it is implemented in the EditDistance.py file. Why is Laplace smoothing needed in order to make the prob method work? In other words, the prob method wouldn’t work properly without smoothing – why?
Laplace smoothing works by adding 1 to all your occurence counts in order to avoid probabilities of 0 - the training set will never be good enough to provide a valid probability for every possibility. It is implemented here:
for a in alphabet:
    for b in alphabet:
        counts[a][b] += .1
By adding .1 to every probability.
The prob method wouldn't work without smoothing because it calls log() on each probability, and log(0) is undefined.

2. Describe the command-line interface for `EditDistance.py`. What command should you run to generate a model from `/data/spelling/wikipedia_misspellings.txt` and save it to `ed.pkl`

The command-line interface has two arguments: `store`, the name of the file to which you'd like to save your model, and `source`, the text source you want to train your model on! As long as `/data/spelling/wikipedia_misspellings.txt` is in the same directory as `EditDistance.py`, then we could run:

`python EditDistance.py --store ed.pkl --source data/spelling/wikipedia_misspellings.txt`

to generate a model from `/data/spelling/wikipedia_misspellings.txt` and save it to `ed.pkl`. 

3. What n-gram orders are supported by the given LanguageModel class?

It supports unigrams and bigrams. 

4. How does the given LanguageModel class deal with the problem of 0-counts?

It lumps all unknown words into a new word, "UNK", for all words unknown to the training model. 

5. What behavior does the `__contains__()` method of the LanguageModel class provide?

Returns true if input is in the vocabulary; false if otherwise. 

6. Spacy uses a lot of memory if it tries to load a very large document. To avoid that problem, LanguageModel limits the amount of text that’s processed at once with the `get_chunks` method. Explain how that method works.

The `get_chunks` method reads in a set of source files and for each file reads in a chunk of that file. The method caps the size of the chunk processed at once with the parameter `chunk_size` which has a default of 100000 (but the user can pass in whatever they like). 

7. Describe the command-line interface for LanguageModel.py. What command should you run to generate a model from /data/gutenberg/*.txt and save it to lm.pkl if you want an alpha value of 0.1 and a vocabulary size of 40000?

The command line interface has two required arguments, source and store, where source is what the model uses to train and store is what it uses to write the model into a file. It also has two optional arguments, alpha and vocab, which have default values, and alpha appears to be used for smoothing and vocabulary to set the allowable size of the vocabulary to collect from the data.

For the above requirements, the command could be:

`python3 LanguageModel.py /data/gutenberg/*.txt --store lm.pkl --alpha 0.1 --vocab 40000`


## Evaluation

1. How often did your spell checker do a better job of correcting than ispell? Conversely, how often did ispell do a better job than your spell checker?

More often than not, ispell outperforms our spell checker, especially where capitalization or punctuation are concerned. 

2. Can you characterize the type of errors your spell checker tended to best at, and the type of errors ispell tended to do best at?

Our spell checker right now isn't checking for capitalization in the middle of a sentence, but ispell preserves the ones that are correct (ie: DNA and China). However, our spell checker does better with removing random amounts of punctuation that shouldn't be there. For instance, ispell leaves "::::Jmabel;" as it is, but our spell checker autocorrects that to "Mabel" (which makes more sense). 

3. Comment on anything else you notice that is interesting about spell checking – either for your model or for ispell.

Our spellchecker autocorrects names it doesn't recognize. This is a commmon thing in autocorrect that most of us technology users are well aware of! However, ispell largely leaves proper names like that alone! Interesting choice. 

## Transpositions

1. Our approach was to simply iterate through the word and swap every pair of characters, checking if the result was in the vocabulary.
Trying to isolate the location of the swap seemed more time and logic intensive than just trying all the pairs, since the more naive approach is a simple for loop. 

2. This addition fixes the case where there is only an edit distance of 1 given, and to fix mistakes like 'heer', 'mistaek', and 'adn' would take two edits. It condenses what would be a deletion and insertion into one edit.

3. We only tackled transpositions that occur next to eachother, for example, heer, not eerh, since transpositions next to eachother seem to be far more common. If letters are swapped elsewhere, that is more easily taken care of by substitutions.
