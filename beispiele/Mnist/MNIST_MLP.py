import numpy as np


from pathlib import Path
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if not Path('mnist_data.csv').exists():
    mnist = fetch_openml('mnist_784', version=1, parser="auto")
    mnist.data.to_csv("mnist_data.csv",index=False)
    mnist.target.to_csv("mnist_target.csv",index=False)
    X, y = mnist.data, mnist.target.astype(int)
else:
    X, y = pd.read_csv('mnist_data.csv'), pd.read_csv('mnist_target.csv').astype(int)


X_raw, y_raw = X.to_numpy(), y.to_numpy().ravel()
scaler = MinMaxScaler()
X_raw = scaler.fit_transform(X_raw)


X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

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
        for e in range(epochs):
            for x, y in zip(train_x, train_y):
                self.backpropagation(x, self.targets[y], lr)

            print(f'Durchlauf #{e + 1} von {epochs} beendet')

    def backpropagation(self, x, target, lr: float = 0.1):
        x = np.transpose(np.array(x,  ndmin = 2))
        t = np.transpose(np.array(target, ndmin = 2))

        """
        hidden_input = np.dot(self.wih, x)
        hidden_output = self.transfer(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.transfer(final_input)
        """
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


def main():
    mlp = MLP(28*28,100,10)
    mlp.fit(X_train,y_train,epochs = 1)
    score, _ = mlp.score(X_test,y_test)
    print(f"Performance: {score:0.1%}" )

if __name__ == "__main__":
    main()

