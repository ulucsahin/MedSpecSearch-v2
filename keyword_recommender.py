from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pickle

import DataLoader

ClassDict = {}
with open('fold0classDict.pkl', 'rb') as f:
    ClassDict = pickle.load(f)

# Better, more personal.
def get_symptom_input_similarity(index, symptoms, softmax_results_symcat, all_results, raw_user_input):
    """
    This method check cosine similarity between user input and symcat symptom 
    descriptions. Offers a more personal keyword recommendation.
    
    Parameters
    ----------
    index : int
        Which keyword to offer.
        0 being most relevant keyword. 1 being second relevant keyword etc.
    symptoms : ndarray or list 
        Symptoms data as string.
    softmax_results_symcat : ndarray
        Softmax results of symcat category descriptions. This is taken from neural network's softmax layer.
    all_results : dict
        results returned from Evaluation of user input (helper.evaluatePerformance method).
    raw_user_input : string
        User input in its raw form.
        
    Returns
    -------
    string
        Recommended keyword.
    """
    user_result = all_results["Scores"][0]
    
    # calculate similarities for with symptoms
    cosine_similarities = []
    for i in range(len(softmax_results_symcat)):
        cosine_similarities.append(cosine_similarity(user_result, softmax_results_symcat[i]))

    cosine_similarities = np.array([a[0][0] for a in cosine_similarities])
    
    # get most similar symptom indexes
    highest_indexes = cosine_similarities.argsort()[:0:-1] 
    
    # we dont want to ask user symptoms that user already asked, so remove them
    # if keyword is present in user data skip it
    highest_relation = list(symptoms[highest_indexes])
    for item in raw_user_input.split(" "):
        for i, keyword in enumerate(highest_relation):
            if item in keyword:
                del highest_relation[i]

    return highest_relation[index]


# Worse, static recommendations according to medical specialty categories.
def get_next_symptom_top3class(index, choice, symptoms, cosine_sim_indexes_all_classes, all_results, raw_user_input):
    """
    This method recommends keywords according to medical specialy categories predicted by user. 
    This method does not take cosine similarity between user input and symptom into consideration.
    
    Parameters
    ----------
    index : int
        Which keyword to offer.
        0 being most relevant keyword. 1 being second relevant keyword etc.
    choice: int
        Choice selects which class we should get keywords from, first second or third, 
        first having highest confidence(best option) while second and third have lower confidence.
    all_results : dict
        results returned from Evaluation of user input (helper.evaluatePerformance method).
    raw_user_input : string
        User input in its raw form.
        
    Returns
    -------
    string
        Recommended keyword.
    """
    # top3 classes for this prediction
    user_top3 = np.array(all_results["Top3"][0])[:,0]
    user_result = all_results["Scores"][0] 
    
    symptom_index = cosine_sim_indexes_all_classes[ClassDict[user_top3[choice]]][index]
    keyword_to_ask = symptoms[symptom_index]
    
    symptom_words = keyword_to_ask.split(' ')
    symptom_word_count = len(symptom_words)
    
    # count how many words user already explained
    included_word_count = 0
    for word in raw_user_input.split(' '):
        if word in symptom_words:
            included_word_count +=1
    
    # if explained words are more than 66% of symptom words then skip to next symptom
    if included_word_count / symptom_word_count > 0.66:
        return get_next_symptom_top3class(index+1, choice, all_results, raw_user_input)
    
    return keyword_to_ask   


def process_user_input(user_input):
    """
    Processes raw user input so that it is ready for evaluation from neural network.
    
    Parameters
    ----------
    user_input : string
        Raw user input in string format.
     
    Returns
    -------
    ndarray
        User input splitted in ndarray format.
    """
    user_input = DataLoader.DataHandler.cleanTextData(user_input)
    user_input = np.array(DataLoader.DataHandler.textIntoWordList(user_input, 128)[0])
    
    return user_input