# Name Classification By Nations

# Summary

A Natural Language Processing application, that classifies names by nations. The model is classic character-level Recurrent Neural Network (RNN). The dataset is taken from original PyTorch manual page, that has names by nations except Turkish names. Turkish names are taken from web, and all names are in uppercase mode. They are converted into lowercase words with capitalizing. 

# Data Preprocessing

All data in .txt format. In data directory, datas are stored in files that have name format 'like nationx.txt'. Filenames have been used for data labeling with string tokenizing operations. 

Then all language characters turned into a one-hotted vector based on defined character-set: 

- "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'i̇şğüçİı"



