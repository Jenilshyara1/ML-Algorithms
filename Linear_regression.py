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
        for _ in range(self.iterations):
            y_pred = self.predict(x_train)
            grad_w = ((2) * (np.dot(y_pred - y_train, x_train))) / self.row
            grad_w0 = np.sum(y_pred - y_train) * (2 / self.row)
            self.W = self.W - self.lr * grad_w
            self.W0 = self.W0 - self.lr * grad_w0

    def predict(self, X: np.ndarray):
        """
        This method predicts the labels for the provided data.

        Parameters:
            x_train (np.ndarray): The data for which to predict labels.

        Returns:
            np.ndarray: The predicted labels for the provided data.
        """
        return np.dot(X, self.W) + self.W0

    def MSE(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        This method calculates the mean squared error of the model's predictions.

        Parameters:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            float: The mean squared error of the model's predictions.
        """
        return np.mean((y_true - y_pred) ** 2)

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        This function calculates the R-squared score, a statistical measure that represents 
        the proportion of the variance for a dependent variable that's explained by an 
        independent variable or variables in a regression model.

        Parameters:
        y_true (numpy array): The ground truth target values.
        y_pred (numpy array): The estimated target values.

        Returns:
        float: The R-squared score. The best possible score is 1.0 and it can be negative 
            (because the model can be arbitrarily worse). A constant model that always 
            predicts the expected value of y, disregarding the input features, would get 
            a R^2 score of 0.0.
        """
        y_mean = np.mean(y_true)
        SSR = np.sum((y_true - y_pred) ** 2)
        SST = np.sum((y_true - y_mean) ** 2)
        return 1 - (SSR / SST)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Linear_regression import LinearRegression
    from sklearn.model_selection import train_test_split
    x = np.random.rand(200,1)
    y = 3.56 * x[:, 0] + 5.11 + np.random.normal(1, 0.1, size=x[:,0].shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    LR = LinearRegression(lr=0.01, iterations=1000)
    LR.fit(x_train, y_train)
    y_pred = LR.predict(x_test)
    score = LR.score(y_test, y_pred)
    print(score)
    plt.scatter(x_test, y_test, color="blue")
    plt.plot(x_test, y_pred, color="red")
    plt.show()
