import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def gallery(images, rows, cols, cmap=None, figsize=(10, 10)):
    #plt.figure(figsize=figsize)
    for n in range(rows * cols):
        plt.subplot(rows, cols, n + 1)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(images[n], cmap)
    plt.show()


def draw_vector(v, o = (0,0), c = 'black'):
    l = np.linalg.norm(v)
    plt.arrow(o[0],o[1],v[0],v[1], head_width=0.05, head_length=0.05, fc='c', ec='c')     

def plot_separator(x1_start,x1_end,w,b):
    try:
        line =  [  [x1_start,x1_end  ], [- (b + w[0] * x1_start) / w[1],- (b + w[0] * x1_end)   / w[1] ]]
        plt.plot(line[0],line[1],c='black')    
    except Exception:
        pass;

def plot_weights(w):
    w_scaled = 0.2 * w / np.linalg.norm(w)
    plt.arrow(0.5,0.5,w_scaled[0],w_scaled[1], head_width=0.05, head_length=0.05, fc='black', ec='black') 

def plot_dots(p,X):

    dots = [255 * p.predict(x) for x in X] 
    plt.scatter(X[:,0],X[:,1],cmap=cm.coolwarm, c = dots);     
    
def plot_perceptron(Y):
    X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
    plt.figure(figsize=(15,10))
    x1_start = x2_start = -0.25
    x1_end = x2_end = 1.25
    density = 0.03    
    input = np.array([[x1,x2] for x1 in np.arange(x1_start,x1_end,density) for x2 in np.arange(x2_start,x2_end,density)])
    
    for n,y in enumerate(Y):
        p = Perzeptron(2)
        p.fit(X,y)        
   
        ax = plt.subplot(2,3,n+1)
        ax.axis([x1_start,x1_end,x1_start,x1_end])
        plt.title(f'y= {y}, w,b = {p.w}, {p.b}')
        
        colors = [p.forward(x) * 255 for x in input]        
        plt.scatter(input[:,0],input[:,1],cmap=cm.coolwarm,c=colors); 

        plot_dots(p,X)        
        plot_separator(x1_start,x1_end,p.w,p.b)
        plot_weights(p.w)

#################################### Laden der MNIST-Daten ##################################

from pathlib import Path
import pandas as pd

from sklearn.datasets import fetch_openml

def load_mnist():
    if not Path('mnist_data.csv').exists():
        mnist = fetch_openml('mnist_784', version=1, parser="auto")
        mnist.data.to_csv("mnist_data.csv",index=False)    
        mnist.target.to_csv("mnist_target.csv",index=False)  
        X, y = mnist.data, mnist.target.astype(int)
    else:
        X, y = pd.read_csv('mnist_data.csv'), pd.read_csv('mnist_target.csv').astype(int)

    return X.to_numpy(), y.to_numpy().ravel()        
        
######################################### PERZEPTRON ##########################################

class Perzeptron:
    def __init__(self,n):
        self.w = np.zeros(n)
        self.b = 0.
    
    def predict(self,x):
        return 1 if self.forward(x) > 0 else 0
    
    def fit(self,x_train,y_train,epsilon=0.01,iter=100):
        for _ in range(iter):
            for x,y in zip(x_train, y_train):
                self.hebb(x,y,epsilon)

    def forward(self, x):
        return np.sum(self.w * x) + self.b        
        
    def hebb(self,x,y,epsilon):
        o = self.predict(x)  
        self.w += epsilon * (y - o) * x
        self.b += epsilon * (y - o)

    def score(self, x, y):
        return np.sum([self.predict(x) == y]) / len(x)

    def __str__(self):
        return f'Gewichte: {self.w}, Bias: {self.b}'


################################# MLP ##########################

import numpy as np

class MLP:
    def __init__(self, n_input, n_hidden, n_output):

        self.wih = 2 * np.random.rand(n_hidden, n_input)  - 1
        self.who = 2 * np.random.rand(n_output, n_hidden) - 1

        self.transfer = lambda x: 1 / (1 + np.exp(-x))
        self.targets = [np.array([int(a == b) for a in range(n_output)], ndmin = 2) for b in range(n_output)]

    def predict(self, x):
        return np.argmax(self.forward(x.T)[0])

    def forward(self, x):
        #x = np.array(x, ndmin = 2).T

        hidden_input = np.dot(self.wih, x)
        hidden_output = self.transfer(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.transfer(final_input)

        return final_output, hidden_output

    def fit(self, train_x, train_y, lr = 0.1, epochs = 1):
        for e in  range(epochs):
            for x, y in zip(train_x, train_y):
                self.backpropagation(x, self.targets[y], lr)

            print(f'Durchlauf #{e + 1} von {epochs} beendet')

    def backpropagation(self, x, target, lr: float = 0.1):
        x = np.transpose(np.array(x, ndmin = 2))
        t = np.transpose(np.array(target, ndmin = 2))

        final_output, hidden_output = self.forward(x)
        output_error = t - final_output
        hidden_error = np.dot(self.who.T, output_error)

        self.who += lr * np.dot((output_error * final_output * (1.0 - final_output)), np.transpose(hidden_output))
        self.wih += lr * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)), np.transpose(x))

    def score(self, test_x, test_y):
        falsch = []
        for n, (x, y) in enumerate(zip(test_x, test_y)):
            if self.predict(x) != y:
                falsch.append(n)
        result = 1. - (len(falsch) / len(test_x))

        return result, falsch

    def save(self,file: str) -> None:
        with open(file + '.npy', 'wb') as f:
            np.save(f,self.wih, allow_pickle=True)
            np.save(f,self.who, allow_pickle=True)
        print("Gewichte wurden gespeichert")            

    def load(self,file: str) -> None:
        with open(file + '.npy', 'rb') as f:
            self.wih = np.load(f)
            self.who = np.load(f)
        print("Gewichte wurden geladen")        
        
    def __str__(self) -> str:
        return "in -> hidden:" + np.array2string(self.wih) +"\nhidden -> out" + np.array2string(self.who) 