# coding: utf-8

# In[1]:


import os
import re
from os.path import join, dirname

import numpy as np
from google.cloud import translate

canTranslate = False
jsonFilePath = join(os.curdir, 'data', 'aut.json')
if os.path.isfile(jsonFilePath):
    canTranslate = True

if canTranslate:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = join(os.curdir, 'data', 'aut.json')


# In[2]:

def checkInputData(inputData):
    """
    Checks if input data is correct or not.

    Parameters
    ----------
    inputData : list or ndarray
        Containing input data.

    Returns
    -------
    bool
        False if there is a problem. True if not.
    """
    charList = [[x] for x in inputData]
    charSet = set(charList)
    uniqChars = len(charSet)

    wordList = inputData.split(" ")
    wordSet = set(wordList)
    uniqWords = len(wordSet)

    if uniqWords < 4 or uniqChars < 7:
        return False
    return True


turkishLabels = ['Deri ve Zührevi Hastalıkları (Cildiye)', 'İç Hastalıkları (Dahiliye)', 'Nöroloji',
                 'Kadın Hastalıkları ve Doğum', 'Göz Hastalıkları', 'Ortopedi ve Travmatoloji',
                 'Kulak Burun Boğaz Hastalıkları', 'Çocuk Sağlığı ve Hastalıkları', 'Ruh Sağlığı ve Hastalıkları',
                 'Radyoloji', 'Genel Cerrahi', 'Üroloji']
englishLabels = ['Dermatology', 'Internal Medicine', 'Neurology', 'Obstetrics & Gynecology', 'Ophthalmology',
                 'Orthopaedic Surgery', 'Otolaryngology', 'Pediatrics', 'Psychiatry',
                 'Radiology-Diagnostic', 'Surgery-General', 'Urology']

labelMatches = zip(turkishLabels, englishLabels)

labelTranslateDict = {y: x for x, y in labelMatches}

print(labelTranslateDict)


# This class handles data operations.
class DataHandler:
    if canTranslate:
        translator = translate.Client()

    turkishLabelsDict = labelTranslateDict

    def __init__(self):
        pass

    @staticmethod
    def getUniqueClassMapDict(classList):
        """
        Assings an integer value to each class in dataset.

        Parameters
        ----------
        classList : ndarray
            List of classes in dataset.

        Returns
        -------
        dict
            Dictionary of Class value pairs.
        """
        uniques = np.unique(classList)
        count = np.arange(uniques.size)
        listDict = np.hstack((uniques.reshape(-1, 1), count.reshape(-1, 1)))
        uniqueDict = {elem[0]: int(elem[1]) for elem in listDict}
        return uniqueDict

    @staticmethod
    def translateInput(inputTR):
        """
        Translates Turkish input to English.

        Parameters
        ----------
        inputTR : string
            Input to be translated to English.

        Returns
        -------
        string
            Text translated to English.
        """
        if not canTranslate:
            raise Exception("Google aut.json file missing")
        return DataHandler.translator.translate(inputTR, target_language="en")["translatedText"]

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def cleanText(text):
        """
        Parameters
        ----------
        textList : list
            Batch of text's

        Returns
        -------
        list
            List of cleaned (removed unwanted symbols or chars) words
        """

        try:
            text = text.lower()
        except:
            print(text)
            # raise exception("oops")
        text = text.replace("Ãƒâ€šÃ‚Â", "")
        text = re.sub(' +', ' ', text)

        cleanText = ""

        for word in text.split(" "):
            cleanWord = ""
            if not DataHandler.is_number(word):
                for char in word:
                    if (ord(char) > 96 and ord(char) < 123):
                        cleanWord += char + ""
                    else:
                        cleanWord += ""
            else:
                cleanWord = word

            cleanText += cleanWord + " "
            cleanText = re.sub(' +', ' ', cleanText)

        return cleanText.strip()


    @staticmethod
    def cleanTextData(textList):
        """

        Parameters
        ----------
        textList : list
            Batch of text's

        Returns
        -------
        list
            List of cleaned (removed unwanted symbols or chars) words
        """

        cleanTextList = []
        for text in textList:
            try:
                text = text.lower()
            except:
                print(text)
                # raise exception("oops")
            text = text.replace("Ãƒâ€šÃ‚Â", "")
            text = re.sub(' +', ' ', text)

            cleanText = ""

            for word in text.split(" "):
                cleanWord = ""
                for char in word:
                    if (ord(char) > 96 and ord(char) < 123):
                        cleanWord += char + ""
                    else:
                        cleanWord += ""

                cleanText += cleanWord + " "
                cleanText = re.sub(' +', ' ', cleanText)
            cleanTextList += [cleanText.strip()]

        return cleanTextList

    @staticmethod
    def idxListToidxDict(idxList):
        """
        Converts list of index to a dictionary.

        Parameters
        ----------
        idxList : list
            List of indexes.

        Returns
        -------
        dict
            List of indexes converted to dictionary.
        """
        idxDict = {}

        for i in range(len(idxList)):
            idxDict[idxList[i]] = i

        return idxDict

    @staticmethod
    def calculateLongestSentence(sentenceList):
        """
        Calculates longest sentence lenght in list of sentences.

        Parameters
        ----------
        sentenceList : list
            List of sentences (strings).

        Returns
        -------
        int
            Length of longest sentence in data.
        """
        longestSentence = 25

        for elem in sentenceList:
            stcLength = len(elem.split(" "))
            if (stcLength > longestSentence):
                longestSentence = stcLength

        return longestSentence

    @staticmethod
    def fillSentenceArray(sentence, fillSize, maxLength=1500):
        """
        Similar to calculateLongestSentence, but works on all data instead
        of just one sentence.

        Parameters
           ----------
           sentence : ndarray
               Numpy array of vectorized words
           fillSize : int
               final array size
           maxLength : int
               final array size

           Returns
           -------
        ndarray
            Numpy array of vectors
        """
        fillCount = fillSize - len(sentence)

        for i in range(fillCount):
            sentence += [np.zeros(300)]

        return sentence[0:maxLength]

    @staticmethod
    def fillWordListArray(sentence, maxLength):
        """
        Pads list with "[None]" strings given word list.
        Makes sentence (list of strings(words)) length equal to
        maxLength.

        Parameters
        ----------
        sentence :  list
            List of strings
        maxLength : int
            Final array size

        Returns
        -------
        list
            List of strings

        """
        fillCount = maxLength - len(sentence)

        for i in range(fillCount):
            sentence += ["[None]"]

        return sentence[0:maxLength]

    @staticmethod
    def textIntoWordList(textList, maxLength, embedModel=None):
        """
        split text into words

        vectorize if embedModel is given

        Parameters
        ----------
        textList : list
            List of strings
        maxLength : int
            Size of final array
        embedModel : string
            (deprecated) Vectorize textList if embed model is given

        Returns
        -------
        List
            List of strings
        int
            Length of sentence used
        """
        embedList = []
        lengthList = []

        for sentence in textList:
            embeddedSentence = []

            for word in sentence.split(" "):
                if (embedModel is not None):
                    if word in embedModel:
                        embedding = word
                        embeddedSentence += [embedding]
                else:
                    embedding = word
                    embeddedSentence += [embedding]

            sentenceLength = len(embeddedSentence)
            embeddedSentence = DataHandler.fillWordListArray(embeddedSentence, maxLength)
            embedList += [embeddedSentence]
            lengthList += [sentenceLength]

        return embedList, sentenceLength

    @staticmethod
    def masterPreprocessor(data, maxLength, shuffle=False, classDict=None):
        """
        Processes data makes it ready for Neural Network.

        Parameters
        ----------
        data : ndarray
            Numpy array of (ndarray of strings(sentences), string) tuple
        maxLength : int
            Max length of sentences to be padded
        shuffle : bool
            Specifies if data should be shuffled or not
        classDict : dict
            Dictionary of labels

        Returns
        -------
        ndarray
            Numpy array of words
        ndarray
            Numpy array of label numbers(for one-hot)
        dict
            Dictionary of classes
        -------
        """
        if (classDict is None):
            classDict = DataHandler.getUniqueClassMapDict(data[:, 1])
        if (shuffle == True):
            np.random.shuffle(data)

        convertedClasses = np.array([classDict[elem] for elem in data[:, 1]])
        print("Outputs converted to numerical forms")
        cleanedTextData = DataHandler.cleanTextData(data[:, 0])
        print("Input text claned")
        wordList, lengthList = DataHandler.textIntoWordList(cleanedTextData, maxLength)
        print("Input text split into tokens and all inputs padded to maximum length")

        return np.array(wordList), np.array(convertedClasses), classDict

    @staticmethod
    def inputPreprocessor(data, maxLength):
        """
        Cleans text and converts it to word list.

        Parameters
        ----------
        data : ndarray
            Numpy array of (ndarray of strings(sentences), string) tuple
        maxLength : int
            Length of sentences

        Returns
        -------
        list
            List of words(string)
        lengthList : int
            Sentence length
        """
        cleanedTextData = DataHandler.cleanTextData(data)
        wordList, lengthList = DataHandler.textIntoWordList(cleanedTextData, maxLength)
        return wordList, lengthList

    @staticmethod
    def batchIterator(data, target, batchSize):
        """
        Iterates over data to create batches.

        Parameters
          ----------
          data : ndarray
              Desired data to split into batches
          target : ndarray
              Targets of data
          batchSize : int
              Size of each batch

          Returns
          -------
          ndarray
              Batch of data as ndarray
          ndarray
              Batch of target as ndarray
        """
        dataSize = data.shape[0]

        while (True):
            randomIdx = np.random.randint(dataSize, size=batchSize)

            yield np.take(data, randomIdx, axis=0), np.take(target, randomIdx)