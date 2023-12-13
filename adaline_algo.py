import numpy as np


class AdalineLinear:
    def __init__(self, learning_rate=0.01, num_epochs=50, addBias=True, mse_threshold=1e-5):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        self.addBias = addBias
        self.mse = 0
        self.mse_threshold = mse_threshold  # Added mse_threshold as an instance variable

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        if self.addBias:
            self.bias = 0
        else:
            self.bias = None

        for _ in range(self.num_epochs):
            # Calculate the linear activation
            linear_activation = np.dot(X, self.weights)
            if self.addBias:
                linear_activation += self.bias
            # Compute the error
            error = y - linear_activation
            self.mse += np.sum(error ** 2)  # Corrected variable name to self.mse

            # Update the weights and bias using gradient descent
            self.weights += self.learning_rate * np.dot(X.T, error)
            if self.addBias:
                self.bias += self.learning_rate * np.sum(error)

            self.mse /= (2 * num_samples)
            print(self.mse)
            if self.mse < self.mse_threshold:
                break

    def predict(self, X):
        linear_activation = np.dot(X, self.weights)
        if self.addBias:
            linear_activation += self.bias
        linear_activation = signum(linear_activation)
        return linear_activation


def signum(x):
    return np.where(x >= 0, 1, -1)  # Using -1 for negative values


def confusion_matrix_adaline(true_labels, predicted_labels):
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
