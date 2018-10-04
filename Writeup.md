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
