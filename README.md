# Name Classification By Nations

# Summary

A Natural Language Processing application, that classifies names by nations. The model is classic character-level Recurrent Neural Network (RNN). The dataset is taken from original PyTorch manual page, that has names by nations except Turkish names. Turkish names are taken from web, and all names are in uppercase mode. They are converted into lowercase words with capitalizing. 

# Data Preprocessing

All data in .txt format. In data directory, datas are stored in files that have name format 'like nationx.txt'. Filenames have been used for data labeling with string tokenizing operations. 

Then all language characters turned into a one-hotted vector based on defined character-set: 

- "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'i̇şğüçİı"

Based on this one-hotted vectors, for example, the one-hotted vector of character "Ş" will be:


         tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0.]])

and the name Şafak, size of <name_length x 1 x size(language)> will be:


         tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], --> Ş

                 [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], --> A

                 [[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], --> F

                 [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], --> A

                 [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]) -->K
                   
                   
All vectors turned into PyTorch tensor. If the user has GPU support with CUDA, they are automatically converted into CUDA arrays in the script.

# Results

Loss based on random sampling: 

<img src="/img/loss.png" alt="drawing" width="500"/>

Normalized comparisons (predictions) via adjacent matrix. diag(matrix) is the correct predictions:

<img src="/img/acc.png" alt="drawing" width="900"/>

# Some Examples Wİth Negative Log-Likelihood Values

         $ predict("Ebubekir Sıddık"):
         > Ebubekir Sıddık
                  (-0.22) Turkish
                  (-1.73) Arabic
                  (-4.09) Irish
                  
         $ predict("Penderecki"):
         > Penderecki
                  (-0.98) Polish
                  (-1.39) Czech
                  (-1.80) Italian
                  
                  
         $ predict("Şafak"):
         > Şafak
                  (-0.82) Czech
                  (-1.53) Turkish
                  (-2.08) Polish
                  
         $ predict("Vivaldi"):
         > Vivaldi
                  (-0.09) Italian
                  (-3.90) French
                  (-4.51) Russian


