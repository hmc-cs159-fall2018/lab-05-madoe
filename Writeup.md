1. In Writeup.md, explain how Laplace smoothing works in general and how it is implemented in the EditDistance.py file. Why is Laplace smoothing needed in order to make the prob method work? In other words, the prob method wouldn’t work properly without smoothing – why?

2. Describe the command-line interface for `EditDistance.py`. What command should you run to generate a model from `/data/spelling/wikipedia_misspellings.txt` and save it to `ed.pkl`

The command-line interface has two arguments: `store`, the name of the file to which you'd like to save your model, and `source`, the text source you want to train your model on! As long as `/data/spelling/wikipedia_misspellings.txt` is in the same directory as `EditDistance.py`, then we could run:

`python EditDistance.py --store ed.pkl --source data/spelling/wikipedia_misspellings.txt`

to generate a model from `/data/spelling/wikipedia_misspellings.txt` and save it to `ed.pkl`. 
