import os
import numpy as np
import pandas as pd
import lrp
import helper
import utility 
import DataLoader

def remove_words(text, important_words):
    """
    Removes words that appear in important words list.

    Parameters
    ----------
    text : ndarray
        ndarray containing splitted sentence.
    important_words : ndarray
        ndarray containing important words. List is suitable as well.

    Returns
    -------
    list
        list containing data after removing words.


    """
    padding_length = len(text)
    text = [a for a in text if a not in important_words]
    while len(text) < padding_length:
        text.append("[None]")
        
    return text

def get_word_imps_all_classes(path, default=True):
    """
    Loads precomputed word importances according to TF-IDF. Eacy category have different index.
    
    Parameters
    ----------
    path : string
        Path to file.
    default : bool
        Specifies if this method should load word importances according to regular TF-IDF words
        or according to stemmed TF-IDF words.
    
    Returns
    -------
    list
        list of (word, importance) tuples where word is string and importance is float.
    """
    dir_ = "default//"
    if not default:
        dir_ = "stemmed//"
        
    files = os.listdir(path + dir_ + "//")
    word_imps_all_classes = []
    for file in files:
        f = open(path + dir_ + "//" + file)
        tmp = []
        for line in f:
            tmp.append(line[0:-1].split(' '))
        tmp = tmp[1:] # remove title
        word_imps_all_classes.append(tmp)
    
    return word_imps_all_classes


def remove_x_tfidf(remove_amount, data, target, word_imps_all_classes):
    """
    Removes most important words from each seperate data instance according to TF-IDF scores.
    This method removes all selected important words. For example, if remove_amount is 1, 
    this method finds the most important word and removes all occurrences of that word from 
    data instance. So it may remove more than 1 word even if remove_amount is 1. 
    
    Parameters
    ----------
    remove_amount : int
        How many important words to select from importance list.
    data : ndarray
        Data to remove words from. Holds ndarrays of string lists.
    target : ndarray
        ndarray of integers (targets).
    word_imps_all_classes : list
        list of (word, importance) tuples where word is string and importance is float.
        
    Returns
    -------
     ndarray
         Data after words removed.
    """
    
    result = []
    total_words_removed = 0
    for i in range(len(data)):
        word_imps = word_imps_all_classes[target[i]]
        most_important_words = np.array(word_imps)[:,0]
        most_important_words = most_important_words[0:remove_amount]
        len1 = 0
        for word in data[i]:
            if word != "[None]":
                len1 += 1
        data_new = np.array([remove_words(a, most_important_words) for a in [data[i]]])[0]
        len2 = 0
        for word in data_new:
            if word != "[None]":
                len2 += 1
        total_words_removed += len1-len2
        result.append(data_new)
    print("total_words_removed", total_words_removed)
    return np.array(result)

    
"""
-------------
EXAMPLE USAGE
-------------

confidence = 0.1
removed_data = remove_x_tfidf(1, testData, testTarget, word_imps_all_classes)
results = helper.evaluatePerformance(configs, nnModel, EmbedModel, sess, removed_data, testTarget, 1, 1-confidence)
results["Accuracy"]
"""



def remove_important_sentences(remove_amount, remove_all, raw_data_df, target, word_imps_all_classes, pad_len):
    """
    Delete sentences from data instead of words according to TF-IDF scores. Just like remove_x_tfidf,
    this method may remove more sentences than specified in remove_amount variable.
    
    Parameters
    ----------
    remove_amount : int
        How many important words to select and delete sentences according to it.
    remove_all : bool
        Specifies if all sentences or just one sentence that contain 
        the important word should be deleted.
    raw_data_df : ndarray
        Data in its raw string form without and preprocess or splitting. raw_data_df holds
        ndarray of strings only.
    target : ndarray
        target of data in ndarray form. It holds ints.
    word_imps_all_classes : list
        list of (word, importance) tuples where word is string and importance is float.
    pad_len : int
        Final data instance length. 
    
    Returns
    -------
    list
        Data after sentences removed. List of strings.
    list
        List holding indexes of data instances with removed sentences.
    """
    result = []
    indexes = [] # indexes in original data of result
    data = raw_data_df #["question"]
    deleted_sentence_count = 0
    for i, item in enumerate(data):   
        tmp = " "
        for word in item.split(' ')[0:pad_len]:
            tmp += word + " "
        # split current item into sentences
        sentences = utility.clean_str(tmp).replace("\\?", "\\.").split("\\.")
        # get truth for current instance
        truth = int(target[i])
        # get important words for current class according to tfidf
        important_words = np.array(word_imps_all_classes[truth][0:remove_amount])[:,0]      
        
        if remove_all:
            # delete all sentences that contains an important word
            tmp_sentences = []
            for j, sentence in enumerate(sentences):
                if not any(word in sentence for word in important_words):
                    tmp_sentences.append(sentence)
                else:
                    indexes.append(i)
                    deleted_sentence_count += 1
            sentences = tmp_sentences
        elif not remove_all:
            # remove only one sentence
            for j, sentence in enumerate(sentences):
                if any(word in sentence for word in important_words):
                    del sentences[j]
                    indexes.append(i)
                    deleted_sentence_count += 1
                    break
            
        # merge sentences to make similar to original data
        new_item = ""
        for sentence in sentences:
            new_item += sentence + " "

        new_item = DataLoader.DataHandler.cleanTextData([new_item])
        new_item, lengthList = DataLoader.DataHandler.textIntoWordList(new_item, pad_len)
    
        result.append(new_item)
        

        if i%20==0:
            print("Completed: %.2f" % (i/len(data)*100), "%", end="\r")
        if i==len(data)-1:
            print("Completed 100.00")
    print("deleted_sentence_count", deleted_sentence_count)
    result = [np.array(a[0]) for a in result]
    return result, indexes


    """
    -------------
    EXAMPLE USAGE
    -------------

    sentences_removed_data, indexes = remove_important_sentences(2, False, testData_raw, testTarget, word_imps_all_classes, 128)
    # sentence_removed_data results all instances
    results = evaluatePerformance(nnModel, sess, np.array(sentences_removed_data), testTarget, 1, 1-confidence)
    results["Accuracy"]
    """

def remove_words_with_index(remove_amount, data, word_imps_unsorted):
    """
    Does not remove all occurrences of important words, but instead removes
    words only in certain indexes. If remove_amount is 1, this method only removes
    one word from each instance.
    
    Parameters
    ----------
    remove_amount : int
        How many words to remove.
    data : ndarray
        Data to remove words from.
    word_imps_unsorted : list
        List of word importances that aligns with words in data. 
        For example first index of word_imps_unsorted is value of 
        first index of data.
    
    Returns
    -------
    list
        Data after words removed.
    """
    pad_len = 128
    removed_data = []
    for i in range(len(data)):
        indexes_to_remove = word_imps_unsorted[i][0:remove_amount][:,2]
        removed_instance = list(np.delete(data[i], indexes_to_remove))

        while(len(removed_instance) < pad_len):
            removed_instance.append("[None]")

        removed_data.append(removed_instance)
    
    return removed_data
    


def get_word_importances(data_raw, EmbedHandler, nnModel, sess, should_save, path=None):
    """
    Create complete list of word importances, each row matching a row in data
    Takes a while, inefficient code.
    
    Parameters
    ----------
    data_raw : ndarray
        Data in ndarray form after sentences are splitted.
    EmbedHandler : EmbeddingHandler
        EmbeddingHandler object. This object should have an embedding model loaded.
    nnModel : CNN
        Models.CNN object.

    Returns
    -------
    list 
        list of word importances
    """
    
    result = []
    for i in range(len(data_raw)):
        data_vectorized = [EmbedHandler.vectorizeSentence(data_raw[i])]
    
        # Get word relevances
        alpha = 1
        layer_count = 1
        weights, biases, activations = helper.get_weights_biases_acts(layer_count, nnModel)
        lrp_layers = lrp.lrp_layers(alpha, layer_count, activations, weights, biases)
        word_importances, _ = lrp.get_word_relevances(alpha, lrp_layers, layer_count, data_vectorized, data_raw[i], sess, nnModel, activations, weights, biases)
#         word_importances.sort(key= lambda x:x[1])
        result.append(word_importances)
    
        print("Completed: %.2f" % (i/len(data_raw)*100), "%", end="\r")

    if should_save :
        np.save(path, result)
    
    return result


def prepare_word_imps(word_importances, testData):
    """
    This method prepares word importance list for remover.remove_words_with_index method.

    Parameters
    ----------
    word_importances : list
        Word importance list generated by remover.get_word_importances method.
    testData : ndarray
        Test data.

    Returns
    -------
    ndarray
        Word importance list sorted after adding indexes. Now we have indexes and we sorted 
        the array according to importances of words.
    """

    # add third dimiension 
    indexes = np.arange(len(testData[0]))

    array_3d = []
    for i in range(len(word_importances)):
        added = np.hstack((word_importances[i].reshape(-1,2), indexes.reshape(-1,1)))
        array_3d.append(added)
    word_importances = array_3d

    array_3d = []
    for i in range(len(word_importances)):
        sorted_array = sorted(word_importances[i], key=lambda x: x[1], reverse=True)
        array_3d.append(sorted_array)
    word_importances = np.array(array_3d)

    return word_importances