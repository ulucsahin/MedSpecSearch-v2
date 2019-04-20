import random
import numpy as np
import re

def unison_shuffled_copies_numpyarray(a, b):
    """
    Unison shuffles numpy arrays and returns as numpy array.

    Parameters
    ----------
    a : ndarray
    b : ndarray

    Returns
    -------
    list
        Shuffled
    list
        Shuffled
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def unison_shuffled_copies_list(a, b):
    """
    Unison shuffles lists and returns.

    Parameters
    ----------
    a : list
    b : list

    Returns
    -------
    list
        Shuffled
    list
        Shuffled
    """
    assert len(a) == len(b)
    zipped_array = list(zip(a, b))
    random.shuffle(zipped_array)
    a, b = zip(*zipped_array)

    return a, np.array(b)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    Parameters
    ----------
    string : string
        String that we want to clean

    Returns
    string 
        Cleaned string

    """
    try:
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\.", " \.", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    except:
        return "unknown"  