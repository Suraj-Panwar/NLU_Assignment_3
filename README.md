# NLU_Assignment_3
This repository contains three program files related to Named Entity Recognition and a brief report about them, the functions of the program are mentioned as follows:

1. Mallet.py
On execution this file creates mutilple files of the same dataset but with different combinations of features which can then be fed to Mallet for tagging. The program is supplemented by a glove.6B.50d.txt file which contains the 50 dimension word embedding of 6 billion words in diferent context and has been used to create embeddings for the words. The file names of created files suggests the features combinations used in the particular file and can be used for feature permuted study.

2. SKLEARN_CRF.py
The file contains the program to execute CRF library implemented in SKLEARN the file input is the data and the program creates a summary of the results obtained during the execution of the program using a 10 fold cross-validation using sklearn library.

3. Neural_NER.py
The file contains the code for executing a LSTM network on the target padded embeddings to produce a probability based model for NER recognition, the program is executed using Keras with tensorflow backend and the resulting accuracy on the validation set is obtained and displayed in the program.
