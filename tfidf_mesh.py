from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from matplotlib import pyplot as plt
from nltk.stem.snowball import SnowballStemmer

import pandas
import numpy as np
import pickle
import DataLoader
import utility
import os
import re
import itertools

configs = {
    "vectorSize":300,
    "trainNewModel":True,
    "dataColumn":"question",
    "maxLength":128,
    "batchSize":8,
    "embeddingType":"asdasdasdasdasd",
    "ELMo":True,
    "PreEmbed":True,
    "restore":True
}

classDict = {'Dermatology': 0,
 'Internal Medicine': 1,
 'Neurology': 2,
 'Obstetrics & Gynecology': 3,
 'Ophthalmology': 4,
 'Orthopaedic Surgery': 5,
 'Otolaryngology': 6,
 'Pediatrics': 7,
 'Psychiatry': 8,
 'Radiology-Diagnostic': 9,
 'Surgery-General': 10,
 'Urology': 11}

fold0ClassDict = classDict

classList = [key for key,value in classDict.items()]

reverse_class_dict = {}
for i in range(len(classList)):
    reverse_class_dict[i] = classList[i]



def sort_coo(coo_matrix):
    """
    Sorts coo_matrix (tfidf_transformer.transform.tocoo) in a reverse fashion.
    
    Example usage:
    from sklearn.feature_extraction.text import TfidfTransformer
    tf_idf_vector = tfidf_transformer.transform(cv.transform([data]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    
    Parameters
    ----------
    coo_matrix : coo_matrix
        coo_matrix that will be sorted. coo_matrix can be obtained 
        from tfidf_transformer.transform.tocoo() method.
    
    Returns
    -------
    tuple
        Reverse sorted tuple of coo_matrix.
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """
    Gets the feature names and TF-IDF score of top n items.
    
    Parameters
    ----------
    feature_names : list
        List containing feature names, obtained from sklearn method
        CountVectorizer.get_feature_names()
    sorted_items : tuple
        Reverse sorted tuple of coo_matrix. Obtained from sort_coo method.
    topn: int
        How many items will be returned from top items.
    
    Returns
    -------
    dict
        Dictionary that holds only topn TF-IDF results.
    """
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def compute_tfidf():
    """ 
    Computes TF-IDF scores by using sort_coo and extract_topn_from_vector methods. 
    Saves results to seperate files for each category.
    This method is not in an OOP fashion.
    """
    vectorizer = TfidfVectorizer()
    transformer = TfidfTransformer()

    docs = []

    for category in classList:
        docs.append(data_dict[category])

    cv=CountVectorizer(max_df=0.85)
    word_count_vector=cv.fit_transform(docs)

    cv=CountVectorizer(max_df=0.85,max_features=10000)
    word_count_vector=cv.fit_transform(docs)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    
    for category in classList:
        feature_names = cv.get_feature_names()
        doc = data_dict[category]
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        #extract only the top n; n here is 10
        keywords = extract_topn_from_vector(feature_names, sorted_items,100)

        f = open("data//icliniq//iCliniq_14K//tfidf_results//" + category + ".txt", "w+", encoding="utf-8")
        f.write("=== Most important words for " + category + " ===\n") 
        for k in keywords:
            f.write(k + " " + str(keywords[k]) + "\n")
        f.close()


def get_word_imps_all_classes(path, default=True):
    """
    Used for loading TF-IDF results of data. Precomputed results are available
    for latest iCliniq data (with 14k data instances).
    
    Parameters
    ----------
    path : String
        Path containing precomputed TF-IDF results.
    
    Returns
    -------
    list
        List containing precomputed TF-IDF results.
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

def get_word_counts(print_ = False):
    """
    Computes and returns common words between TF-IDF results of each category for iCliniq_14K data.
    
    Parameters
    ----------
    print_ : bool
        Specifies if results should be printed or not.
        
    Returns
    -------
    ndarray
        Results array containing common words between 
        TF-IDF results of each category for iCliniq_14K data.
    """
    ClassDict = {}
    with open('fold0classDict.pkl', 'rb') as f:
        ClassDict = pickle.load(f)
    outputSize = len(ClassDict)
    
    word_imps_all_classes = get_word_imps_all_classes("data//icliniq//iCliniq_14K//tfidf_results//", True)
    reverse_class_dict = {}
    for item in ClassDict:
        reverse_class_dict[ClassDict[item]] = item
        
    results = np.zeros((12,12))
    for i in range(12):
        for j in range(12):
            list1 = np.array(word_imps_all_classes[i])[:,0][0:20]
            list2 = np.array(word_imps_all_classes[j])[:,0][0:20]
            commons = len(set(list1) & set(list2))
            results[i,j] = commons
            if print_:
                a = reverse_class_dict[i]
                b = reverse_class_dict[j]
                print("Common word amount between {} and {} is {}".format(a,b,commons))
            
    return results

def plot_confusion_matrix(cm, classes, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Parameters
    ----------
    cm : ndarray 
        Results Matrix to be plotted. Should be a 2D numpy array.
    classes : list
        List of strings that contain class names.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    _ , pltAx = plt.subplots(figsize=(20, 9))
    cmap=plt.cm.Blues
    title='Confusion matrix'
    pltAx.imshow(cm, interpolation='nearest', cmap=cmap)
    #pltAx.colorbar()
    tick_marks = np.arange(len(classes))
    pltAx.set_xticks(tick_marks )
    pltAx.set_yticks(tick_marks )
    
    pltAx.set_xticklabels(classes,rotation=75)
    pltAx.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pltAx.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.show()


# Symcat 
def clean(string):   
    """
    Cleans string and makes it suitable for Symcat operations.
    
    Parameters
    ----------
    string : str
        String to be cleaned.
    
    Returns
    -------
    str
        Cleaned string.
    """

    string = string.replace("e.g.", "")
    string = string.replace("i.e.", "")
    string = string.replace("&quot;", "")
    return string

# Symcat
def merge(text_array):
    """
    Merges list of strings into one string.
    
    Parameters
    ----------
    text_array : list
        List of strings that will be merged.
    
    Returns
    -------
    list
        Merged list.
    """
    result = ""
    
    for item in text_array:
        result += item + " "
    
    result = result[0:-1]
    
    return result

def setup_symcat():
    # Read Symcat data
    symcat = []
    with open("data//symcat//symcat_data.txt", "r") as f:
        for line in f:
            symcat.append(line)
    f.close()

    # Get categories
    categories = []
    for item in symcat:
        category = re.search('\"\>(.*)\<\/a', item)
        if category:
            result = str(category.group(1)).lower()
            categories.append(result)
    del categories[49] # error in data, gets aligned with description after this delete operation

    # Get Descriptions
    descriptions = []
    for item in symcat:
        description = re.search('\>\\t\\t\\t(.*)\\n', item)
        if description:
            result = str(description.group(1)).lower()
            descriptions.append(result)

    descriptions = [clean(a) for a in descriptions]

    # All tfidf words are merged for categories
    word_imps_all_classes = get_word_imps_all_classes("data//icliniq//iCliniq_14K//tfidf_results//", True)
    merged_tfidf = [merge(np.array(a)[:,0]) for a in word_imps_all_classes]

    # Description + Categories in one list
    desc_plus_cat = []
    for i in range(len(categories)):
        desc_plus_cat.append(categories[i] + descriptions[i])
    
    
    categories = np.array(categories)
    descriptions = np.array(descriptions)

    tfidf = TfidfVectorizer().fit_transform(desc_plus_cat + merged_tfidf)
    cosine_similarities = linear_kernel(tfidf[473:485], tfidf[0:473]) # 12 classes 

    
    return cosine_similarities, descriptions

def get_most_relevant_symcat(category, descriptions, cos_similarities):
    """
    Returns the most relevant symcat categories for selected category.
    
    Parameters
    ----------
    category : int
        Category to display results for.
    descriptions : ndarray
        Numpy array of containing strings of symcat descriptions.
    cos_similarities : ndarray
        Numpy array of containing cosine similarities between symcat datas and 
        TF-IDF words of 12 classes.
    
    Returns
    -------
    ndarray
        Numpy array of containing most relevant 6 classes for selected category
    """
    
    cos_sim0 = cos_similarities[category].flatten()
    indexes = cos_sim0.argsort()[:-6:-1]
    descriptions[indexes] # these are the most relevant ones with class 0
    
    return descriptions

def get_data_dict(trainData_raw, trainTarget, testData_raw, testTarget, reverse_class_dict):
    """
    Splits icliniq data into categories (converts it to dictionary).

    Parameters
    ----------
    trainData_raw : ndarray
        Hold raw train data before being processed.
    trainTarget : ndarray
        Holds train labels.
    testData_raw : ndarray
        Hold raw test data before being processed.
    testTarget : ndarray
        Holds test labels.
    reverse_class_dict : dict
        Holds int-string key-value pairs for class labels.

    Returns
    -------
    dict
        Data splitted into classes in dictionary.
    """    
    data_dict = {}

    for i, row in enumerate(trainData_raw):
        target = reverse_class_dict[trainTarget[i]]
        try:
            data_dict[target] = data_dict[target] + " " + row
        except:
            data_dict[target] = row
        print(i/len(trainData_raw)*100, end ="\r")
    print("train data finished")

    for i, row in enumerate(testData_raw):
        target = reverse_class_dict[testTarget[i]]
        try:
            data_dict[target] = data_dict[target] + " " + row
        except:
            data_dict[target] = row
        print(i/len(testData_raw)*100, end ="\r")
    print("test data finished")

    return data_dict