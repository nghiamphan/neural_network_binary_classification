import math
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import utils as U

from sklearn.model_selection import train_test_split


# Neural Network with 0 hidden layer
class NeuralNetwork:
    def __init__(self):
        return None

    def import_data(
        self,
        file_name: str = None,
        file_header: int = None,
        data_frame: pd.DataFrame = None,
        test_size: int = 0.2,
        normalization: bool = True,
    ):
        """
        Import data by file or dataframe.
        """
        if file_name != None:
            df = pd.read_csv(file_name, header=file_header)
        else:
            df = data_frame

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        self.setup_nn(normalization)

    def setup_nn(self, normalization: bool = False):
        """
        Normalize the inputs if necessary. Add the bias node for each input vector. Set up weights.
        """
        if normalization:
            U.normalize_data(self.X_train, self.X_test)
        print(self.X_train)
        print(self.X_test)

        bias = [[1]] * len(self.X_train)
        self.X_train = np.hstack((bias, self.X_train))
        self.y_train = np.array(self.y_train)

        bias = [[1]] * len(self.X_test)
        self.X_test = np.hstack((bias, self.X_test))
        self.y_test = np.array(self.y_test)

        self.setup_weight()

    def setup_weight(self):
        self.w = np.random.rand(len(self.X_train[0]))
        self.original_w = np.array(self.w)

    def reset_weight(self):
        self.w = np.array(self.original_w)

    def sigmoid(self, X: list[float] | list[list], w: list[float]) -> float | list[float]:
        """
        Calculate sigmoid function.

        If X is an 1-d array (representing a single input vector),  return a single value.
        If X is a  2-d array (representing multiple input vectors), return a list of floats.
        """
        z = X @ w
        if X.ndim == 1:
            return 1 / (1 + math.exp(z))
        return np.array([1 / (1 + math.exp(n)) for n in z])

    def forward_propagation(self, X: list[float] | list[list]) -> float | list[float]:
        return self.sigmoid(X, self.w)

    def backward_propagation(self, X: list[float] | list[list], y: int | list[int], l2: float) -> list[float]:
        """
        Parameters
        ----------
        X: list | list[list]
            A single input vector | Multiple input vectors
        y: int | list
            A single target value | A list of target values
        l2: float
            Lambda factor in L2 regularization

        Return
        ------
        dw: list[float]
            List of the gradients of the weights between input and output layers
        """
        y_predict = self.forward_propagation(X)

        if X.ndim == 1:
            dw = X * (y_predict - y) - l2 * self.w
        else:
            dw = X.T @ (y_predict - y) / len(X) - l2 * self.w

        return dw

    def quick_train(
        self,
        X: list[list],
        y: list[int],
        epochs: int = 100,
        learning_rate: float = 1,
        l2: float = 0,
    ) -> tuple[float, list[float]]:
        """
        Train the neural network and update weights after the forward pass of each input vector.

        Parameters
        ----------
        X: list[list]
            Training data input
        y: list[int]
            Training data target
        epochs: int
            Number of training iterations
        learning_rate: float
        l2: float
            Lambda factor in L2 regularization
        print_result: bool
            Whether to print some result after each training iteration

        Return
        -------
        test_log_loss: float
            Log loss on test data at the end of the training
        scores: list[train_accuracy, train_log_loss, test_accuracy, test_log_loss]
            list of scores for each epoch
        """
        scores = []
        self.reset_weight()
        for epoch in range(1, epochs + 1):
            for i in range(len(X)):
                dw = self.backward_propagation(X[i], y[i], l2)
                self.w += learning_rate * dw

            train_accuracy, train_log_loss = self.evaluate(X, y)
            test_accuracy, test_log_loss = self.evaluate(self.X_test, self.y_test)

            self.print_result(epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, self.w)
            scores.append([train_accuracy, train_log_loss, test_accuracy, test_log_loss])

        return test_log_loss, scores

    def slow_train(
        self,
        X: list[list],
        y: list[int],
        epochs: int = 100,
        learning_rate: float = 1,
        l2: float = 0,
    ) -> tuple[float, list[float]]:
        """
        Train the neural network and update weights after each entire epoch forward pass.

        Parameters & Return
        -------------------
        Same as method quick_train()
        """
        scores = []
        self.reset_weight()
        for epoch in range(1, epochs + 1):
            dw = self.backward_propagation(X, y, l2)
            self.w += learning_rate * dw

            train_accuracy, train_log_loss = self.evaluate(X, y)
            test_accuracy, test_log_loss = self.evaluate(self.X_test, self.y_test)

            self.print_result(epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, self.w)
            scores.append([train_accuracy, train_log_loss, test_accuracy, test_log_loss])

        return test_log_loss, scores

    def cross_validation(self, k_fold: int, epochs: int, learning_rate: float, l2: float):
        """
        Train the neural network with cross validation and report the average log loss.

        Parameters
        ----------
        k_fold:
            Number of folds
        epochs: int
            Number of training iterations
        learning_rate: float
        l2: float
            Lambda factor in L2 regularization
        """
        test_log_loss = 0
        for i in range(k_fold):
            n = len(self.X_train) // k_fold

            X = np.vstack((self.X_train[: i * n], self.X_train[(i + 1) * n :]))
            y = np.hstack((self.y_train[: i * n], self.y_train[(i + 1) * n :]))

            test_log_loss += self.quick_train(X, y, epochs, learning_rate, l2)

        test_log_loss /= k_fold

        print(
            f"Epochs: {epochs} {'Learning Rate:':>20} {learning_rate} {'R2 Lambda:':>15} {l2:<6} {'Test Log Loss:':>25} {test_log_loss}"
        )

    def predict(self, X: list[list]) -> list[int]:
        y_probability = self.forward_propagation(X)

        # transform predicted probabilities into predicted class labels
        y_predict = [0 if prob < 0.5 else 1 for prob in y_probability]

        return y_probability, y_predict

    def evaluate(self, X: list[list], y: list[int]) -> tuple[float, float]:
        y_probability, y_predict = self.predict(X)
        accuracy = metrics.accuracy_score(y, y_predict)
        log_loss = metrics.log_loss(y, y_probability)
        self.classification_report = metrics.classification_report(y, y_predict)
        self.confusion_matrix = metrics.confusion_matrix(y, y_predict)
        return accuracy, log_loss

    def draw_confusion_matrix(self):
        figure = plt.figure(figsize=(10, 5))

        ax = sns.heatmap(
            self.confusion_matrix,
            annot=True,
            cmap="YlGnBu",
            fmt="g",
            xticklabels=["negative", "positive"],
            yticklabels=["negative", "positive"],
        )
        ax.tick_params(length=0, labeltop=True, labelbottom=False)
        ax.xaxis.set_label_position("top")

        plt.title("Confusion matrix", y=1.12, fontsize=18)
        plt.xlabel("Predicted label", fontsize=13)
        plt.ylabel("Actual label", fontsize=13)
        plt.show()

        return figure

    def print_result(
        self,
        epoch: int,
        train_accuracy: float,
        train_log_loss: float,
        test_accuracy: float,
        test_log_loss: float,
        *args,
    ):
        """
        Print accuracy and log loss by epoch.
        """
        print(
            f"Epoch {epoch:>3}: {'Train accuracy':>18} {train_accuracy:8.4f} {'Train loss:':>15} {train_log_loss:8.4f} {'Test accuracy':>18} {test_accuracy:8.4f} {'Test loss:':>12} {test_log_loss:2.4f} {'Weights:':>15}",
            end=" ",
        )
        for arg in args:
            print(arg, end=" ")

        print()

    def draw_result(self, scores: list[list]):
        """
        Plot accuracy and log loss by epoch.
        """
        scores = np.array(scores)

        figure = plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Accuracy")
        plt.plot(scores[:, 0], label="Train", color="tab:blue")
        plt.plot(scores[:, 2], label="Test", color="tab:green")
        plt.xlabel("Epoch")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.title("Log Loss")
        plt.plot(scores[:, 1], label="Train", color="tab:blue")
        plt.plot(scores[:, 3], label="Test", color="tab:green")
        plt.xlabel("Epoch")
        plt.legend(loc="lower left")
        plt.show()

        return figure


# Neural Network with 1 hidden layer
class NeuralNetworkHidden(NeuralNetwork):
    def __init__(self, n_hidden_nodes):
        """
        The hidden layer contains n_hidden_nodes and one bias node
        """
        self.n_hidden_nodes = n_hidden_nodes

    def setup_weight(self):
        """
        W1: list[list]
            Matrix of weights between input and hidden layers
        w2: list[float]
            List of weights between hidden and output layers
        """
        self.W1 = np.random.rand(len(self.X_train[0]), self.n_hidden_nodes)
        self.w2 = np.random.rand(self.n_hidden_nodes + 1)

        self.original_W1 = np.array(self.W1)
        self.original_w2 = np.array(self.w2)

    def reset_weight(self):
        """
        This function is called before each training in cross validation to make comparison between different parameters fairer.
        """
        self.W1 = np.array(self.original_W1)
        self.w2 = np.array(self.original_w2)

    def leaky_relu(self, X: list[float] | list[list], w: list[list]) -> list[float] | list[list]:
        """
        Calculate leaky RELU function: f(z) = z if z >= 0 and f(z) = 0.01*z if z < 0.

        If X is an 1-d array (representing a single input vector),  return a list representing a hidden vector.
        If X is a  2-d array (representing multiple input vectors), return a list of list representing a list of hidden vectors.
        """
        z = X @ w
        for n in np.nditer(z, op_flags=["readwrite"]):
            if n < 0:
                n[...] *= 0.01
        return z

    def forward_propagation(self, X: list[float] | list[list]) -> float | list[float]:
        """
        If X is an 1-d array (representing a single input vector),  return a single value representing the final output.
        If X is a  2-d array (representing multiple input vectors), return a list of floats representing a list of corresponding final outputs.
        """
        self.h_1dim = np.ones(self.n_hidden_nodes + 1)
        self.h_2dim = np.ones((len(X), self.n_hidden_nodes + 1))

        if X.ndim == 1:
            # hidden layer
            self.h_1dim[1:] = self.leaky_relu(X, self.W1)
            # output layer
            return self.sigmoid(self.h_1dim, self.w2)
        else:
            # hidden layer
            self.h_2dim[:, 1:] = self.leaky_relu(X, self.W1)
            # output layer
            return self.sigmoid(self.h_2dim, self.w2)

    def backward_propagation(
        self, X: list[float] | list[list], y: int | list[int], l2_1: float, l2_2: float
    ) -> tuple[list[float], list[list]]:
        """
        Parameters
        ----------
        X: list | list[list]
            A single input vector | Multiple input vectors
        y: int | list
            A single target value | A list of target values
        l2_1: float
            Lambda factor in L2 regularization of the weights between input and hidden layers
        l2_2: float
            Lambda factor in L2 regularization of the weights between hidden and output layers

        Return
        ------
        dw2: list[float]
            List of the gradients of the weights between hidden and output layers
        DW1: list[list]
            Matrix of the gradients of the weights between input and hidden layers
        """
        y_predict = self.forward_propagation(X)

        if X.ndim == 1:
            dw2 = (y_predict - y) * self.h_1dim - l2_2 * self.w2

            # derivative of leaky RELU
            dh_dz = np.array([0.01 if z < 0 else 1 for z in self.h_1dim[1:]])
            DW1 = (y_predict - y) * (
                np.reshape(X, (len(X), 1)) @ np.reshape(self.w2[1:] * dh_dz, (1, self.n_hidden_nodes))
            ) - l2_1 * self.W1

        else:
            dw2 = self.h_2dim.T @ (y_predict - y) / len(X) - l2_2 * self.w2

            DH_DZ = self.h_2dim[:, 1:]
            for n in np.nditer(DH_DZ, op_flags=["readwrite"]):
                if n < 0:
                    n[...] = 0.01
                else:
                    n[...] = 1

            DW1 = (
                X.T
                @ (np.reshape((y_predict - y), (len(X), 1)) @ np.reshape(self.w2[1:], (1, self.n_hidden_nodes)) * DH_DZ)
                / len(X)
                - l2_1 * self.W1
            )

        return dw2, DW1

    def quick_train(
        self,
        X: list[list],
        y: list[int],
        epochs: int = 100,
        learning_rate_1: float = 1,
        learning_rate_2: float = 1,
        l2_1: float = 0,
        l2_2: float = 0,
    ) -> tuple[float, list[float]]:
        """
        Train the neural network and update weights after the forward pass of each input vector.

        Parameters
        ----------
        X: list[list]
            Training data input
        y: list[int]
            Training data target
        epochs: int
            Number of training iterations
        l2_1: float
            Lambda factor in L2 regularization of the weights between input and hidden layers
        l2_2: float
            Lambda factor in L2 regularization of the weights between hidden and output layers
        print_result: bool
            Whether to print some result after each training iteration

        Return
        -------
        test_log_loss: float
            Log loss on test data at the end of the training
        scores: list[train_accuracy, train_log_loss, test_accuracy, test_log_loss]
            list of scores for each epoch
        """
        scores = []
        self.reset_weight()
        for epoch in range(1, epochs + 1):
            for i in range(len(X)):
                dw2, DW1 = self.backward_propagation(X[i], y[i], l2_1, l2_2)
                self.w2 += learning_rate_2 * dw2
                self.W1 += learning_rate_1 * DW1

            train_accuracy, train_log_loss = self.evaluate(X, y)
            test_accuracy, test_log_loss = self.evaluate(self.X_test, self.y_test)

            self.print_result(epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, self.W1[0], self.w2)
            scores.append([train_accuracy, train_log_loss, test_accuracy, test_log_loss])

        return test_log_loss, scores

    def slow_train(
        self,
        X: list[list],
        y: list[int],
        epochs: int = 100,
        learning_rate_1: float = 1,
        learning_rate_2: float = 1,
        l2_1: float = 0,
        l2_2: float = 0,
    ) -> tuple[float, list[float]]:
        """
        Train the neural network and update weights after each entire epoch forward pass.

        Parameters & Return
        -------------------
        Same as method quick_train()
        """
        scores = []
        self.reset_weight()
        for epoch in range(1, epochs + 1):
            dw2, DW1 = self.backward_propagation(X, y, l2_1, l2_2)
            self.w2 += learning_rate_2 * dw2
            self.W1 += learning_rate_1 * DW1

            train_accuracy, train_log_loss = self.evaluate(X, y)
            test_accuracy, test_log_loss = self.evaluate(self.X_test, self.y_test)

            self.print_result(epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, self.W1[0], self.w2)
            scores.append([train_accuracy, train_log_loss, test_accuracy, test_log_loss])

        return test_log_loss, scores

    def cross_validation(
        self,
        k_fold: int,
        epochs: int,
        learning_rate_1: float,
        learning_rate_2: float,
        l2_1: float,
        l2_2: float,
    ):
        """
        Train the neural network with cross validation and report the average log loss.
        """
        test_log_loss = 0
        for i in range(k_fold):
            n = len(self.X_train) // k_fold

            X = np.vstack((self.X_train[: i * n], self.X_train[(i + 1) * n :]))
            y = np.hstack((self.y_train[: i * n], self.y_train[(i + 1) * n :]))

            test_log_loss += self.quick_train(X, y, epochs, learning_rate_1, learning_rate_2, l2_1, l2_2)

        test_log_loss /= k_fold

        print(
            f"Epochs: {epochs} {'Hidden Layer Learning Rate:':>30} {learning_rate_1} {'Output Layer Learning Rate:':>30} {learning_rate_2} {'Hidden Layer R2 Lambda:':>30} {l2_1:<6} {'Output Layer R2 Lambda:':>30} {l2_2:<6} {'Test Log Loss:':>30} {test_log_loss}"
        )


SAMPLE_FILE = "data_banknote_authentication.txt"

if __name__ == "__main__":
    nn = NeuralNetworkHidden(10)
    nn.import_data(file_name=SAMPLE_FILE, file_header=None, normalization=True)
    test_log_loss, scores = nn.slow_train(nn.X_train, nn.y_train, epochs=50, l2_1=0.01, l2_2=0.01)

    # nn = NeuralNetwork()
    # nn.import_data(file_name=SAMPLE_FILE, file_header=None, normalization=False)
    # test_log_loss, scores = nn.slow_train(nn.X_train, nn.y_train, epochs=100)

    nn.draw_result(scores)
