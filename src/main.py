from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
"""
@author: safak
"""

def findFiles(path): return glob.glob(path)

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def nameToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def getLabel(output):
    top_n, top_i = output.topk(1)
    """
    -> Returns the k largest elements of the given input tensor along a given dimension.
    """
    label_i = top_i[0].item()
    return labels[label_i], label_i

"""
we pick random names and its labels from training data
and we train our model with this random choice
this two randomChoice() and randomTrainingExample() functions do that.
"""
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    label = randomChoice(labels)
    name = randomChoice(category_lines[label])
    label_tensor = torch.tensor([labels.index(label)], dtype=torch.long)
    name_tensor = nameToTensor(name)
    return label, name, label_tensor, name_tensor


def evaluate(name_tensor):
    hidden = model.initHidden()
    if CUDA:
        hidden = hidden.cuda()
    for i in range(name_tensor.size()[0]):
        output, hidden = model.forward(name_tensor[i], hidden)

    return output

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc2 = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.fc1(combined)
        out = self.fc2(combined)
        out = self.softmax(out)
        return out, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


if __name__ == "__main__":
    
    all_letters = string.ascii_letters + " .,;'i̇şğüçİı"
    n_letters = len(all_letters)
    
    print(findFiles('/media/safak/Data/Desktop HDD/Deep Learning/PyTorch/RNN/data/names/*.txt'))
    print("\n")
    print(all_letters)
    print(n_letters)
    print(unicodeToAscii('Şafak BİLİCİ'))
    print(unicodeToAscii("Sıddık Çağrı Özsütoğulları"))
    
    category_lines = {}
    labels = []
    
    for filename in findFiles('/media/safak/Data/Desktop HDD/Deep Learning/PyTorch/RNN/data/names/*.txt'):
        #print((os.path.basename(filename))) #filenames
        #print(os.path.splitext(os.path.basename(filename))) # split filenames (filename,'.extension')
        label = os.path.splitext(os.path.basename(filename))[0]
        #print(label) 
        labels.append(label)
        lines = readLines(filename)
        category_lines[label] = lines
        
    nlabels = len(labels)
    print("Number of nations: ",nlabels)
    
    print("Tensor Representation of Letter 'Ş': ",letterToTensor('Ş'))
    print("Tensor Size of Name 'Şafak': ",nameToTensor('Şafak').size()) # (line_length x 1 x n_letters),
    
    
    n_hidden = 128  
    model = RNN(n_letters, n_hidden, nlabels) ##model
    
    # One-hotted vectors
    input = letterToTensor('A')
    hidden =torch.zeros(1, n_hidden)
    
    output, hidden = model.forward(input, hidden)
    
    input = nameToTensor('Şafak')
    hidden = torch.zeros(1, n_hidden)
    
    output, next_hidden = model.forward(input[0], hidden)
    print(output)
    
    """
    print(lineToTensor('Şafak'))               tensor(
        [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
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
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]) --> K
    """


    """
    Building the model
    """
    criterion = nn.NLLLoss()
    lr = 0.005
    iters = 100000
    print_every = 5000
    plot_every = 1000
    current_loss = 0
    loss_history = []
    
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()
    """
    Training Loop
    """
    for iter in range(1,iters+1):
        label, name, label_tensor, name_tensor = randomTrainingExample()
        if CUDA:
            label_tensor = label_tensor.cuda()
            name_tensor = name_tensor.cuda()
        #print(label)
        #print(label_tensor)
        #rint(name)
        #print(name_tensor)
   
        hidden = model.initHidden()
        if CUDA:
            hidden = hidden.cuda()
        model.zero_grad()
        for i in range(name_tensor.size()[0]):
            output, hidden = model.forward(name_tensor[i], hidden)
        
        loss = criterion(output, label_tensor)
        loss.backward()

        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-lr)
        
        current_loss += loss
        if iter % plot_every == 0:
            print("iter/loss: {}/{} ".format(iter,current_loss / plot_every))
            loss_history.append(current_loss / plot_every)
            current_loss = 0
        
    plt.figure()
    plt.plot(loss_history)
    
    cm = torch.zeros(nlabels,nlabels)
    ncm = 10000
    
    for i in range(ncm):
        label, name, label_tensor, name_tensor = randomTrainingExample()
        if CUDA:
            label_tensor = label_tensor.cuda()
            name_tensor = name_tensor.cuda()
            
        output = evaluate(name_tensor)
        guess, guess_i = getLabel(output)
        label_i = labels.index(label)
        cm[label_i][guess_i] += 1
        
    for i in range(nlabels):
        cm[i] = cm[i] / cm[i].sum()
        
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm.numpy())
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels, rotation=90)
    ax.set_yticklabels([''] + labels)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    
#%%
    def predict(name, n_predictions=3):
        print('\n> %s' % name)
        with torch.no_grad():
            output = evaluate(nameToTensor(name).cuda())
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []
        
            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, labels[category_index]))
                predictions.append([value, labels[category_index]])
    
    
    predict("Berk")


