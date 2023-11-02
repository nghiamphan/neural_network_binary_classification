import pandas as pd
import matplotlib.pyplot as plt


def normalize_column(column: pd.Series, min_val: float = None, max_val: float = None) -> object:
    """
    Normalize data according to a given min and max value.

    Parameters
    ----------
    df: pd.Series
    min_val: float
    max_val: float

    Return
    ------
    df: pd.Series
        data after normalization
    """
    if min_val == None:
        min_val = column.min()

    if max_val == None:
        max_val = column.max()

    column = 2 * (column - min_val) / (max_val - min_val) - 1
    return column


def normalize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """
    Normalize the training data between -1 and 1 for each column.
    Normalize the test data according to the min and max value of each column in training data.

    Parameters
    ----------
    X_train: pd.DataFrame
        training set
    X_test: pd.DataFrame
        test set
    """
    for i in range(len(X_train.iloc[0])):
        min_val = X_train.iloc[:, i].min()
        max_val = X_train.iloc[:, i].max()
        X_train.iloc[:, i] = normalize_column(X_train.iloc[:, i])
        X_test.iloc[:, i] = normalize_column(X_test.iloc[:, i], min_val, max_val)

    return None


def box_plot(df: pd.DataFrame):
    """
    Draw box plot of a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
    """
    figure = plt.figure(figsize=(10, 5))
    plt.boxplot(df, labels=df.columns)

    return figure
