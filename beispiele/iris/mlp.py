
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