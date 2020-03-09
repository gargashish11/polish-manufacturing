from typing import TypeVar
from typing import Optional
from pathlib import Path
import joblib as jl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures


def index_class_split(dataframe, col_names, index=None):
    '''
    splits a dataframe in two parts based on the index and target column
    name.One part, will have all the columns except those specified by
    col_names and the other part will have the col_names.
    Default setting of index=None,will return all rows of the dataframe.
    Parameters
    ----------
    dataframe : pandas dataframe.
    input dataframe
    col_names : string or list of string
    index : int or list of ints, optional

    Returns
    -------
    X : dataframe
        dataframe slice with all columns except those specified in col_names
    y : dataframe
        dataframe slice with all columns specified in col_names.

    '''
    dataframe = dataframe.loc[:].reset_index(drop=True)
    if index == None:
        index = dataframe.index
    if type(col_names) == str:
        col_names = [col_names]
    X = dataframe.loc[index, ~dataframe.columns.isin(col_names)]
    y = dataframe.loc[index, dataframe.columns.isin(col_names)]
    return X, y


def savefunc_asobj(filename, func, ignore_file=False, **kwargs):
    fname = Path(filename)
    if fname.exists() and not ignore_file:
        obj = jl.load(filename)
    else:
        obj = func(**kwargs)
        jl.dump(obj, fname, compress=True)
    return obj


def saveexpr_asobj(filename, expression_obj, globals, locals, ignore_file=False):
    fname = Path(filename)
    if fname.exists() and not ignore_file:
        obj = jl.load(filename)
    else:
        obj = eval(expression_obj, globals, locals)
        jl.dump(obj, fname, compress=True)
    return obj


def output_title(input_str, char="-", print_len=79):
    char_len = print_len - len(input_str)
    if char_len >= 2:
        char_len_left = np.int(char_len/2)
        char_len_right = np.int(char_len - char_len_left)
        print(char_len_left*char + input_str + char_len_right*char)
    else:
        raise ValueError(f"input string is longer than {print_len}")


def plot_roc_curve(fpr, tpr, linewidth=2, label=None):
    plt.plot(fpr, tpr, linewidth=linewidth, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()


def wrap_kendalltau(X, y):
    X = np.array(X).transpose()
    y = np.tile(np.ravel(y), (X.shape[0], 1))
    corr, p_value = zip(*[kendalltau(a, b, nan_policy='raise')
                          for a, b in zip(X, y)])
    return corr, p_value


def wrap1(X, y):
    corr = []
    p_value = []
    X = np.array(X).transpose()
    y = np.ravel(y)
    for col in X:
        ktau = kendalltau(col, y, nan_policy='raise')
        corr.append(ktau[0])
        p_value.append(ktau[1])
    return corr, p_value


def wrap2(X, y):
    X = np.asarray(X).transpose()
    y = np.ravel(y)
    cols = X.shape[0]
    corr = np.zeros(cols)
    p_value = np.zeros(cols)
    for i in range(cols):
        #         corr[i], p_value[i] = kendalltau(X[i,:], y, nan_policy='raise')
        ktau = kendalltau(X[i, :], y, nan_policy='raise')
        corr[i] = ktau[0]
        p_value[i] = ktau[1]
    return corr, p_value


def add_interactions(df, comb_len=2, degree=2, interaction_only=True, include_bias=False):
    # Get feature names
    combos = list(combinations(list(df.columns), comb_len))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]

    # Find interactions
    poly = PolynomialFeatures(interaction_only=interaction_only,
                              include_bias=include_bias)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    # # Remove interaction terms with all 0 values
    # noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    # df = df.drop(df.columns[noint_indicies], axis=1)
    return df


def test(filename, func, exec_only=False, ignore_file=False, **kwargs):
    if exec_only:
        obj = func(**kwargs)
    else:
        fname = Path(filename)
        if fname.exists() and not ignore_file:
            obj = jl.load(filename)
        else:
            obj = func(**kwargs)
            jl.dump(obj, fname, compress=True)
    return obj


def test2(filename, func, exec_only=False, ignore_file=False, **kwargs):
    fname = Path(filename)
    if fname.exists() and not ignore_file and not exec_only:
        obj = jl.load(filename)
    else:
        obj = func(**kwargs)
        if not exec_only:
            jl.dump(obj, fname, compress=True)
    return obj


def load_pickle(file_name: str) -> Optional[object]:
    """Load a pickled object from file_name IFF the file exists."""
    return jl.load(file_name) if Path(file_name).exists() else None


def save_pickle(obj: object, file_name: str) -> None:
    """Pickle and save the object to file_name."""
    jl.dump(obj, file_name, compress=True)
