from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter
import tkinter as tk

from sklearn.preprocessing import MinMaxScaler

from PercAlgo import PerceptronAlgo, confusion_matrix_perc
from adaline_algo import AdalineLinear, confusion_matrix_adaline


def activation_function(X):
    return X


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc


def plot_decision_boundary(X, y, weights, bias):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    ws= weights
    print(ws,bias)
    if bias is not None:
        x2 = (-weights[0] * x1 - bias) / weights[1]
    else:
        x2 = (-weights[0] * x1) / weights[1] # No bias term
    plt.plot(x1, x2, color='red', label='Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


def preprocess_data(df):
    X = df[[feature_1_variable.get(), feature_2_variable.get()]]
    class1 = classe_1_Entry.get()
    class2 = classe_2_Entry.get()

    class_mapping = {"BOMBAY": 0, "CALI": 1, "SIRA": 2}

    if class1 != class2 and class1 in class_mapping and class2 in class_mapping:
        c1, c2 = class_mapping[class1], class_mapping[class2]
        X_c1 = X[df["Class"] == c1].sample(frac=1, random_state=42)
        X_c2 = X[df["Class"] == c2].sample(frac=1, random_state=42)

        return X_c1, X_c2
    else:
        print("Invalid class values. Make sure to choose valid classes from BOMBAY, CALI, and SIRA.")
        return None, None


def selected_algo(X_train, bias_value, y_test, y_train):
    try:
        learning_rate = float(LR_ENTRY.get())
    except ValueError:
        learning_rate = 0.001  # Set a default value for learning rate

    try:
        epochs_num = int(EpochsNum_ENTRY.get())
    except ValueError:
        epochs_num = 1000

    selected_algorithm = algos.get()
    y_test = np.where(y_test == 0, -1, 1)
    mse = mse_threshold.get()

    if selected_algorithm == 0:
        p = PerceptronAlgo(addBias=bias_value, learning_rate=learning_rate, iterations_num=epochs_num,)
        p.fit(X_train, y_train)

    elif selected_algorithm == 1:  # Adaline
        adaline = AdalineLinear(learning_rate, epochs_num,bias_value,mse)
        adaline.fit(X_train, y_train)
    return p if selected_algorithm == 0 else adaline, selected_algorithm, y_test


def split_data(X_c1, X_c2, bias_value):
    # Split the data into training and testing sets
    X_train_c1 = X_c1.iloc[:30]
    X_test_c1 = X_c1.iloc[30:50]

    X_train_c2 = X_c2.iloc[:30]
    X_test_c2 = X_c2.iloc[30:50]

    y_train_c1 = df.loc[X_train_c1.index, "Class"]
    y_train_c2 = df.loc[X_train_c2.index, "Class"]
    y_test_c1 = df.loc[X_test_c1.index, "Class"]
    y_test_c2 = df.loc[X_test_c2.index, "Class"]

    X_train = pd.concat([X_train_c1, X_train_c2], axis=0).to_numpy()
    y_train = pd.concat([y_train_c1, y_train_c2], axis=0).to_numpy()
    X_test = pd.concat([X_test_c1, X_test_c2], axis=0).to_numpy()
    y_test = pd.concat([y_test_c1, y_test_c2], axis=0).to_numpy()

    combined_train = np.column_stack((X_train, y_train))

    # Combine X_test and y_test
    combined_test = np.column_stack((X_test, y_test))

    # Shuffle the combined arrays
    np.random.shuffle(combined_train)
    np.random.shuffle(combined_test)

    # Split back into X_train, y_train, X_test, and y_test
    X_train = combined_train[:, :-1]
    y_train = combined_train[:, -1]
    X_test = combined_test[:, :-1]
    y_test = combined_test[:, -1]

    selectedAlgo, selected_algorithm, y_test = selected_algo(X_train, bias_value, y_test, y_train)

    return selectedAlgo, X_train, y_train, X_test, y_test


def logic():
    global df, y_test_combined, y_train_combined, X_test_combined, X_train_combined

    # selected data classes
    X_c1, X_c2 = preprocess_data(df)

    if X_c1 is not None and X_c2 is not None:
        bias_value = bias_var.get()
        p, X_train, y_train, X_test, y_test = split_data(X_c1, X_c2, bias_value)
        predictions = p.predict(X_test)

        selected_algorithm = algos.get()

        if selected_algorithm == 0:
            print(confusion_matrix_perc(y_test, predictions))
        else:
            print(confusion_matrix_adaline(y_test, predictions))

        print("Perceptron classification accuracy: ", accuracy(y_test, predictions))
        weights = p.weights
        bias = p.bias
        plot_decision_boundary(X_train, y_train, weights, bias)

    else:
        print("Class1 and Class2 should be different.")


def filter_classes_gui():
    global window, frame1, classe_1_Entry, classe_2_Entry, FEATURE_1_LABEL, FEATURE_2_LABEL, bias_var, LR_ENTRY, EpochsNum_ENTRY, algos, mse_threshold
    window = tk.Tk()
    window.title("Dry Beans")
    frame = tk.Frame(window)
    frame.pack()
    frame1 = tk.LabelFrame(frame)
    frame1.grid(row=0, column=0, padx=20, pady=20)

    # Entries for learning rate, number of epochs and MSE threshold
    LR_LABEL = tkinter.Label(frame1, text="Enter Learning Rate")
    LR_LABEL.grid(row=0, column=0)
    EpochsNum_LABEL = tkinter.Label(frame1, text="Enter Number Of Epochs")
    EpochsNum_LABEL.grid(row=1, column=0)
    MSEThresh_LABEL = tkinter.Label(frame1, text="Enter MSE Threshold")
    MSEThresh_LABEL.grid(row=2, column=0)
    LR_ENTRY = tkinter.Entry(frame1)
    EpochsNum_ENTRY = tkinter.Entry(frame1)
    mse_threshold = tkinter.DoubleVar()

    MSEThresh_ENTRY = tkinter.Entry(frame1,textvariable=mse_threshold)
    LR_ENTRY.grid(row=0, column=1)
    EpochsNum_ENTRY.grid(row=1, column=1)
    MSEThresh_ENTRY.grid(row=2, column=1)

    bias_var = tkinter.IntVar()  # Create an IntVar to store the value

    # Create the BiasCheck Checkbutton with the IntVar
    BiasCheck = tkinter.Checkbutton(frame1, text="Bias", variable=bias_var)
    BiasCheck.grid(row=8, column=0)

    # To get the value of the BiasCheck Checkbutton:
    algos = tk.IntVar()

    # used algo
    rad1 = tkinter.Radiobutton(frame1, text="Perceptron", value=0, variable=algos)
    rad2 = tkinter.Radiobutton(frame1, text="Adaline", value=1, variable=algos)
    rad1.grid(row=9, column=0)
    rad2.grid(row=9, column=1)

    # Entries for used classes and features
    CLASSE_1_LABEL = tk.Label(frame1, text="first class: ")
    CLASSE_2_LABEL = tk.Label(frame1, text="second class: ")

    classe_1_Entry = ttk.Combobox(frame1, values=["BOMBAY", "SIRA", "CALI"])
    classe_1_Entry.set("BOMBAY")

    classe_2_Entry = ttk.Combobox(frame1, values=["BOMBAY", "SIRA", "CALI"])
    classe_2_Entry.set("CALI")  # Set the default value

    CLASSE_1_LABEL.grid(row=3, column=0)
    classe_1_Entry.grid(row=3, column=1)
    CLASSE_2_LABEL.grid(row=4, column=0)
    classe_2_Entry.grid(row=4, column=1)


def filter_features_gui():
    global feature_1_variable, feature_2_variable
    # Create dropdown menus for selecting feature columns
    FEATURE_1_LABEL = tk.Label(frame1, text="first feature: ")
    FEATURE_2_LABEL = tk.Label(frame1, text="second feature: ")

    available_columns = df.columns.tolist()
    feature_1_variable = tk.StringVar()
    feature_2_variable = tk.StringVar()
    feature_1_variable.set(available_columns[0] if available_columns else "")
    feature_2_variable.set(available_columns[1] if available_columns and len(available_columns) > 1 else "")
    feature_1_dropdown = tk.OptionMenu(frame1, feature_1_variable, *available_columns)
    feature_2_dropdown = tk.OptionMenu(frame1, feature_2_variable, *available_columns)
    FEATURE_1_LABEL.grid(row=5, column=0)
    feature_1_dropdown.grid(row=5, column=1)
    FEATURE_2_LABEL.grid(row=6, column=0)
    feature_2_dropdown.grid(row=6, column=1)
    window.mainloop()


def preprocessing():
    global df  # Define df globally
    df = pd.read_excel('Dry_Bean_Dataset.xlsx')
    execute_button = tk.Button(frame1, text="Execute", command=logic)
    execute_button.grid(row=7, column=0, columnspan=2)
    average_bombay = df.loc[df["Class"] == "BOMBAY", "MinorAxisLength"].mean()
    df["MinorAxisLength"].fillna(average_bombay, inplace=True)
    df["Class"].replace({"BOMBAY": 0, "CALI": 1, "SIRA": 2}, inplace=True)
    df["Class"] = df["Class"].astype(np.int64)
    features = df[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to your data and transform the data
    normalized_features = scaler.fit_transform(features)

    # Replace the original columns with the normalized data
    df[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]] = normalized_features


filter_classes_gui()
preprocessing()
filter_features_gui()
