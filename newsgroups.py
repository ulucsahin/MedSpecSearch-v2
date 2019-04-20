from sklearn.datasets import fetch_20newsgroups
import utility
import pandas as pd
import numpy as np 
import pickle

def get_newsgroups():
    """
    Fetches and processes 20newsgroups data.

    Returns
    -------
    data_train : ndarray
        Processed training data.
    target_train : ndarray
        Training target.
    data_test : ndarray
        Processed test data.
    target_test : ndarray
        Test target.
    class_dict : dict
        Dictionary that maps classes to integers.
    reverse_class_dict : dict
        Dictionary that maps integers to classes.
    """
    
    # Fetch data
    raw_data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    target_names = raw_data_train.target_names
    raw_data_train = pd.DataFrame([raw_data_train.data, raw_data_train.target.tolist()]).T
    raw_data_train.columns = ['text', 'target']

    raw_data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    raw_data_test = pd.DataFrame([raw_data_test.data, raw_data_test.target.tolist()]).T
    raw_data_test.columns = ['text', 'target']

    # Convert data to list from dataframe
    data_train = raw_data_train["text"]
    target_train = raw_data_train["target"]
    data_test = raw_data_test["text"]
    target_test = raw_data_test["target"]
    
    # create class dict and reverse class dict
    class_dict = {}
    for i, item in enumerate(target_names):
        class_dict[item] = i

    reverse_class_dict = {}
    for i, item in enumerate(class_dict):
        reverse_class_dict[i] = item
    
    
    # Clean strings
    # Split sentences by spaces and pad it to 128 length, if longer than 128 cut it to 128
    data_train = [utility.clean_str(a).split(" ") for a in data_train]
    data_test = [utility.clean_str(a).split(" ") for a in data_test]

    for i in range(len(data_train)):
        if len(data_train[i]) > 128:
            data_train[i] = data_train[i][0:128]
        else:
            while len(data_train[i]) < 128:
                data_train[i].append("[None]")


    for i in range(len(data_test)):
        if len(data_test[i]) > 128:
            data_test[i] = data_test[i][0:128]
        else:
            while len(data_test[i]) < 128:
                data_test[i].append("[None]")
                
    
    # Check max length, should be 128
    max_len = 0
    for item in data_train+data_test:
        if len(item) > max_len:
            max_len = len(item)

    assert(max_len == 128)
    
    return data_train, target_train, data_test, target_test, class_dict, reverse_class_dict

def process_newsgroups_numpy(should_fetch, should_save, convert_labels, remove_fq, sentence_lenght=128):
    """
    Maybe Fetches and processes 20newsgroups data into numpy array.
    Makes it ready to give into Neural Net Model.
    Use either this method or get_newsgroups method. There is no need to use both at once.

    Parameters
    ----------
    should_fetch : bool
        Specifies if it should fetch data again or use a save one.
    should_save : bool
        Specifies if it should save data into disk.
    convert_labels : bool
        Specifies it it should convert numerical labels to names.
    remove_fq : bool
        Specifies if it should remove footers and quotes from data.
    sentence_lenght : int
        Specifies max sentence length in each data instance.

    Returns
    -------
    data_train : ndarray
        Processed training data.
    target_train : ndarray
        Training target.
    data_test : ndarray
        Processed test data.
    target_test : ndarray
        Test target.
    class_dict : dict
        Dictionary that maps classes to integers.
    reverse_class_dict : dict
        Dictionary that maps integers to classes.
    """
    
    if should_fetch:
        # Fetch data
        if remove_fq:
            raw_data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            raw_data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        else:
            raw_data_train = fetch_20newsgroups(subset='train', remove=('headers'))
            raw_data_test = fetch_20newsgroups(subset='test', remove=('headers'))

        target_names = raw_data_train.target_names
        raw_data_train = pd.DataFrame([raw_data_train.data, raw_data_train.target.tolist()]).T
        raw_data_train.columns = ['text', 'target']

        raw_data_test = pd.DataFrame([raw_data_test.data, raw_data_test.target.tolist()]).T
        raw_data_test.columns = ['text', 'target']

    if should_save:
        raw_data_train.to_csv("data//20_newsgroups//train.csv")
        raw_data_test.to_csv("data//20_newsgroups//test.csv")
    
    # Read data
    trainData = pd.read_csv("data//20_newsgroups//train.csv")
    testData = pd.read_csv("data//20_newsgroups//test.csv")
    trainData = trainData.values
    testData = testData.values

    # Remove extra dimension
    trainData[:,0] = trainData[:,2]
    trainData = trainData[:,1:3]
    testData[:,0] = testData[:,2]
    testData = testData[:,1:3]
    
    # create class dict and reverse class dict
    class_dict = {}
    for i, item in enumerate(target_names):
        class_dict[item] = i

    reverse_class_dict = {}
    for i, item in enumerate(class_dict):
        reverse_class_dict[i] = item

    classList = [key for key,value in class_dict.items()] #????

    with open('fold0classDict.pkl', 'wb') as f:
        pickle.dump(class_dict, f, pickle.HIGHEST_PROTOCOL)
        
    if convert_labels:
        # Convert numerical labels to names
        trainData[:,1] = [reverse_class_dict[a] for a in trainData[:,1]]
        testData[:,1] = [reverse_class_dict[a] for a in testData[:,1]]

    # Clean text
    trainData[:,0] = [utility.clean_str(a).split(" ") for a in trainData[:,0]]
    testData[:,0] = [utility.clean_str(a).split(" ") for a in testData[:,0]]

    # Convert insides to np array
    trainData[:,0] = [np.asarray(a) for a in trainData[:,0]]
    testData[:,0] = [np.asarray(a) for a in testData[:,0]]

    # Pad or cut to 128
    for i in range(len(trainData)):
        if len(trainData[i][0]) > sentence_lenght:
            trainData[i][0] = trainData[i][0][0:sentence_lenght]
        else:
            while len(trainData[i][0]) < sentence_lenght:
                trainData[i][0]= np.append(trainData[i][0], "[None]")

    for i in range(len(testData)):
        if len(testData[i][0]) > sentence_lenght:
            testData[i][0] = testData[i][0][0:sentence_lenght]
        else:
            while len(testData[i][0]) < sentence_lenght:
                testData[i][0]= np.append(testData[i][0], "[None]")

    
    trainTarget = trainData[:,1]
    trainData = trainData[:,0]
    
    testTarget = testData[:,1]
    testData = testData[:,0]
    
    return trainData, trainTarget, testData, testTarget, class_dict, reverse_class_dict