import numpy as np


class LinearRegression:
    def __init__(self, lr, iterations) -> None:
        self.lr = lr
        self.W: np.ndarray = None
        self.W0: np.ndarray = None
        self.iterations = iterations
        self.row = None

    def fit(self, x_train: np.ndarray, y_train):
        self.row, dim = x_train.shape
        self.W = np.random.rand(dim)
        self.W0 = 0
        for i in range(self.iterations):
            # Wt * x + W0
            y_pred = self.predict(x_train)
            # (2/n)*(y_pred — y)*x
            grad_w = ((2) * (np.dot(y_pred - y_train, x_train))) / self.row
            # (2/n)*sum(y_pred — y)
            grad_w0 = np.sum(y_pred - y_train) * (2 / self.row)
            # w = w — lr*dw
            # w0 = w0 — lr*dw0
            self.W = self.W - self.lr * grad_w
            self.W0 = self.W0 - self.lr * grad_w0

    def predict(self, x_train: np.ndarray):
        return np.dot(x_train, self.W) + self.W0

    def score(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    x_train = np.random.rand(10, 2)
    y_train = 3 * x_train[:, 0] + 4 * x_train[:, 1]
    lr = LinearRegression(0.01, 10)
    lr.fit(x_train, y_train)
    print(lr.W)
    print(lr.W0)
