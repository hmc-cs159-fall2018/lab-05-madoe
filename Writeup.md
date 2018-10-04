1. In Writeup.md, explain how Laplace smoothing works in general and how it is implemented in the EditDistance.py file. Why is Laplace smoothing needed in order to make the prob method work? In other words, the prob method wouldn’t work properly without smoothing – why?

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

