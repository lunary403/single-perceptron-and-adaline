import numpy as np


def signum(x):
    return np.where(x >= 0, 1, -1)


class PerceptronAlgo:
    def __init__(self, learning_rate=0.01, iterations_num=1000, addBias=True):
        self.learning_rate = learning_rate
        self.iterations_num = iterations_num
        self.addBias = addBias
        self.weights = None
        self.bias = 0 if self.addBias else None  # Initialize bias based on 'addBias'

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        if self.addBias:
            self.bias = 0  # Initialize bias if 'addBias' is True

        for _ in range(self.iterations_num):
            for x_i, target in zip(X, y):
                linear_output = np.dot(x_i, self.weights) + self.bias if self.addBias else np.dot(x_i, self.weights)
                y_predicted = signum(linear_output)
                update = self.learning_rate * (target - y_predicted)
                self.weights += update * x_i
                if self.addBias:
                    self.bias += update  # Update bias if 'addBias' is True

    def predict(self, X):
        if self.addBias:
            linear_output = np.dot(X, self.weights) + self.bias
        else:
            linear_output = np.dot(X, self.weights)
        y_predicted = signum(linear_output)
        return y_predicted


def confusion_matrix_perc(true_labels, predicted_labels):
    # Initialize variables for the confusion matrix
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label == -1 and predicted_label == -1:
            true_negative += 1
        elif true_label == -1 and predicted_label == 1:
            false_positive += 1
        elif true_label == 1 and predicted_label == -1:
            false_negative += 1

    # Create the confusion matrix as a dictionary
    confusion_matrix = {
        'True Positive': true_positive,
        'True Negative': true_negative,
        'False Positive': false_positive,
        'False Negative': false_negative
    }

    return confusion_matrix
