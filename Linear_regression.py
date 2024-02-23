import numpy as np

class LinearRegression:
    """
    This class implements a simple linear regression model.

    Attributes:
        lr (float): The learning rate for the model.
        iterations (int): The number of iterations for the model to learn.
        W (np.ndarray): The weights of the model.
        W0 (np.ndarray): The bias of the model.
        row (int): The number of rows in the training data.
    """

    def __init__(self, lr, iterations) -> None:
        """
        The constructor for the LinearRegression class.

        Parameters:
            lr (float): The learning rate for the model.
            iterations (int): The number of iterations for the model to learn.
        """
        self.lr = lr
        self.W: np.ndarray = None
        self.W0: np.ndarray = None
        self.iterations = iterations
        self.row = None

    def fit(self, x_train: np.ndarray, y_train):
        """
        This method trains the model using the provided training data.

        Parameters:
            x_train (np.ndarray): The training data.
            y_train (np.ndarray): The labels for the training data.
        """
        self.row, dim = x_train.shape
        self.W = np.random.rand(dim)
        self.W0 = 0
        for i in range(self.iterations):
            y_pred = self.predict(x_train)
            grad_w = ((2) * (np.dot(y_pred - y_train, x_train))) / self.row
            grad_w0 = np.sum(y_pred - y_train) * (2 / self.row)
            self.W = self.W - self.lr * grad_w
            self.W0 = self.W0 - self.lr * grad_w0

    def predict(self, x_train: np.ndarray):
        """
        This method predicts the labels for the provided data.

        Parameters:
            x_train (np.ndarray): The data for which to predict labels.

        Returns:
            np.ndarray: The predicted labels for the provided data.
        """
        return np.dot(x_train, self.W) + self.W0

    def score(self, y_true, y_pred):
        """
        This method calculates the mean squared error of the model's predictions.

        Parameters:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            float: The mean squared error of the model's predictions.
        """
        return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    x_train = np.random.rand(10, 2)
    y_train = 3 * x_train[:, 0] + 4 * x_train[:, 1]
    lr = LinearRegression(0.01, 10)
    lr.fit(x_train, y_train)
    print(lr.W)
    print(lr.W0)
