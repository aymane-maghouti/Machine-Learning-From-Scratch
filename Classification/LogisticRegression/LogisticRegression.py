import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

class LogisticRegression():
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()

        for _ in range(self.n_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_predictions)
        predictions = np.where(y_pred < 0.5, 0, 1)
        return predictions
