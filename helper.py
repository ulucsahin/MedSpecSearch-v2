import numpy as np
import tensorflow as tf
import pickle
import os

# Our code
import DataLoader
import Models

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

classList = [key for key,value in classDict.items()]

reverse_class_dict = {}
for i in range(len(classList)):
    reverse_class_dict[i] = classList[i]

def getTokenLengths(token):
    """
    A utility method that helps getting number of tokens (words) 
    in each data instance that is available in batch.
    
    For example: With batch size 4 and sentence length 128,
    this method would return the list [128, 128, 128, 128]
    
    Parameters
    ----------
    token : ndarray
        Vectorized data to get token lengths (2D Numpy array).
    
    Returns
    -------
    list 
        List of integers containing token lengths
    """
    return [len(item) for item in token]


def evaluatePerformance(configs, nnModel, embedModel, sess,testData,testTarget,batchSize,uncertaintyCoef):
    """
    Method for evaluating performance (accuracy) of trained model.
    
    Parameters
    ----------
    configs: dict
        Dictionary that holds configuration information
    nnModel : CNN
        Model that will be evaluated.
    embedModel : EmbeddingHandler
        EmbeddingHandler object.  
    sess : tf.InteractiveSession
        Currently active Tensorflow session.
    testData : ndarray
        Test data that will be used in evaluating (2D Numpy Array).
    testTarget : ndarray
        Test target that will be used in evaluating (1D Numpy Array).
    batchSize : int
        Batch size that will be used during the evaluation.
    uncertaintyCoef : float
        Uncertainty coefficient. Will be used while getting results
        such as amount of data that exceeds preferred confidence ratio. Note that 
        confidence is 1-uncertaintyCoef.
    
    Returns
    -------
    - outputs : Dictionary
        Results in dictionary form.
    """

    ClassDict = {}
    with open('fold0classDict.pkl', 'rb') as f:
        ClassDict = pickle.load(f)
    outputSize = len(ClassDict)

    reverseClassDict = {value:key for key,value in ClassDict.items()}
    top3 = []
    
    dataSize = testData.shape[0]
    start = 0
    end = batchSize
    
    totalAcc = 0
    totalUcAcc = 0
    totalDataRate = 0
    
    truth = None
    predu = None
    
    testTruth = np.array([])
    testPred = np.array([])
    testScores = []
    
    testEvTrue = 0
    testEvFail = 0
    
    while(start<dataSize):
        data = np.array(testData[start:end])
        dataClean = data
        
        if(configs["PreEmbed"]):
            data = embedModel.vectorizeBatch(data)
        
        outputData = np.array(testTarget[start:end])
        cutSize = data.shape[0]
        tokens_length = getTokenLengths(data)
        
        fd = {nnModel.nn_inputs:dataClean,nnModel.nn_vector_inputs:data,nnModel.nn_outputs:outputData,nnModel.isTraining:False,nnModel.token_lengths:tokens_length,
             nnModel.uncertaintyRatio:uncertaintyCoef}
        
        scores, prob, testBAcc,nnTruth,nnPrediction,nnMatch,evCor,evFail,ucAcc,dataRate = sess.run([nnModel.scores, nnModel.prob, nnModel.accuracy,nnModel.truths,nnModel.predictions
                                                                       ,nnModel.correct_predictions,nnModel.mean_ev_succ,nnModel.mean_ev_fail,nnModel.ucAccuracy,
                                                                                     nnModel.dataRatio]
                                                                      ,feed_dict=fd)
        # For top 3
        prob = prob[0]
        probDict = {reverseClassDict[i]:prob[i] for i in np.arange(outputSize)}
        probMatrix = []
        for i in range(len(prob)):
            probMatrix.append([reverseClassDict[i], prob[i]])
        probMatrix = sorted(probMatrix, key=lambda x: (x[1]), reverse=True)
        top3.append(probMatrix[0:3])
        
        testTruth = np.append(testTruth,nnTruth,axis=0)
        testPred = np.append(testPred,nnPrediction,axis=0)
        testScores.append(scores)
        testEvTrue += evCor*cutSize
        testEvFail += evFail*cutSize 
        
        totalAcc += testBAcc*cutSize
        totalUcAcc += ucAcc*cutSize
        totalDataRate += dataRate*cutSize
        start += batchSize
        end += batchSize
        
    outputs = {
        "Accuracy":totalAcc/dataSize,
        "TotalEvidenceTrue":testEvTrue/dataSize,
        "TotalEvidenceFalse":testEvFail/dataSize,
        "UncertaintyAccuracy":totalUcAcc/dataSize,
        "DataRate":totalDataRate/dataSize,
        "Truth":testTruth,
        "Prediction":testPred,
        "Scores":testScores,
        "Top3":top3
    }
        
    return outputs
    #return (totalAcc/dataSize,testTruth,testPred,testEvTrue/dataSize,testEvFail/dataSize,totalUcAcc/dataSize,totalDataRate/dataSize)

def trainModel(sess, embedModel, nnModel, iterations, trainData, trainTarget, testData, testTarget, configs, accList):
    """
    Method for training a model.
    
    Parameters
    ----------
    sess : InteractiveSession
        Tensorflow session.
    embedModel : EmbeddingHandler
        EmbeddingHandler object.
    nnModel : CNN
        Model that will be trained.
    iterations : int
        Desired number of training iterations.
    trainData : 2D Numpy Array
        Training data that will be used in training the model.
    testTarget : 1D Numpy Array
        Training target that will be used in training the model.
    testData : 2D Numpy Array
        Test data that will be used during the evaluation of model between iterations.
        Mainly used for displaying results during training and plotting the results.
    testTarget : 1D Numpy Array
        Test target that will be used during the evaluation of model between iterations.
    configs : Dictionary
        Dictionary that holds required configuration information. 
    accList: List
        A list that holds results. Currently not used, can be used in displaying results
        or debugging.
    """
    batcher = DataLoader.DataHandler.batchIterator(trainData, trainTarget, configs["batchSize"])
    sample,_ = next(batcher)
    
    print("trainData shape : ", trainData.shape)
    print("testData shape : ", testData.shape)
    print("trainTarget shape : ", trainTarget.shape)
    print("testTarget shape : ", testTarget.shape)
    
    htTestAcc=0
    fold0TestAcc = 0
    ucAcc = 0
    dataRate = 0
    
    L_test_ev_s=[]
    L_test_ev_f=[]
    
    print("")
    for i in range(iterations):
        data, target = next(batcher)
        dataClean = data

        if(configs["PreEmbed"]):
            data = embedModel.vectorizeBatch(data)

        tokens_length = getTokenLengths(data)
        fd = {nnModel.nn_inputs:dataClean, nnModel.nn_vector_inputs:data,nnModel.nn_outputs:target,
              nnModel.isTraining:True,nnModel.token_lengths:tokens_length,nnModel.annealing_step: 100}
        _, acc, los = sess.run([nnModel.train_op, nnModel.accuracy, nnModel.loss], feed_dict=fd)

        if i % 5 == 0:
            title = ("[Current iteration = "+str(i)+" Train Acc:"+str(acc) + " fold0Test: "+str(fold0TestAcc)+"]")
            title += " "*50
            title = str(title)       
            print(title, end="\r")

        if i % 500 == 0 and i != 0:
            oldTestAcc = fold0TestAcc
            testOutputs = evaluatePerformance(configs, nnModel, embedModel, sess, testData, testTarget, configs["batchSize"], 0.1)
            
            fold0TestAcc = testOutputs["Accuracy"]
            fEvTrue = testOutputs["TotalEvidenceTrue"]
            fEvFail = testOutputs["TotalEvidenceFalse"]
            ucAcc = testOutputs["UncertaintyAccuracy"]
            dataRate = testOutputs["DataRate"]
            fTruth = testOutputs["Truth"]
            fPrediction = testOutputs["Prediction"]
            
            # confidences = [0.995,0.98,0.90,0.70,0.5]
            # confidenceMatrix = np.zeros(shape=[len(confidences),3])
            # for idx in range(len(confidences)):
            #     testOutputs = evaluatePerformance(nnModel, sess, testData, testTarget, configs["batchSize"],1-confidences[idx])
            #     confidenceMatrix[idx,0] = confidences[idx]
            #     confidenceMatrix[idx,1] = testOutputs["DataRate"]
            #     confidenceMatrix[idx,2] = testOutputs["UncertaintyAccuracy"]
            
            L_test_ev_s.append(fEvTrue)
            L_test_ev_f.append(fEvFail)
            
            if(fold0TestAcc>oldTestAcc):
                pass
                #saveSession(sess)

            accList.append([i, acc, htTestAcc, fold0TestAcc, los, ucAcc])
            npAccList = np.array(accList)           


def execute_training(should_load, embedModel, iterations, trainData, trainTarget, testData, testTarget, configs,
                     model_path=None):
    """
    Executes whole training operations from scratch. Can be considered as helper method.
    
    Parameters
    ----------
    should_load : bool
        Specifies if a pre-trained model should be loaded or not.
    iterations : int
        Number of training iterations.
    trainData : 2D Numpy Array
        Training data that will be used in training the model.
    testTarget : 1D Numpy Array
        Training target that will be used in training the model.
    testData : 2D Numpy Array
        Test data that will be used during the evaluation of model between iterations.
        Mainly used for displaying results during training and plotting the results.
    testTarget : 1D Numpy Array
        Test target that will be used during the evaluation of model between iterations.
    configs : Dictionary
        Dictionary that holds required configuration information. 
    model_path : string
        Path to model that will be loaded if should_load is True.
        
    Returns
    -------
    InteractiveSession
        Current tensorflow session.
    """
    should_load = should_load
    inputSize = configs["maxLength"]
    vectorSize = configs["vectorSize"]

    ClassDict = {}
    with open('fold0classDict.pkl', 'rb') as f:
        ClassDict = pickle.load(f)
    outputSize = len(ClassDict)

    if configs["model_type"] == "CNN":
        nnModel = Models.CNN(inputSize=inputSize, vectorSize=vectorSize, outputSize=outputSize)
    elif configs["model_type"] == "CNN_3Layer":
        nnModel = Models.CNN_3Layer(inputSize=inputSize, vectorSize=vectorSize, outputSize=outputSize)
    elif configs["model_type"] == "RNN_LSTM":
        nnModel = Models.RNN_LSTM(inputSize=inputSize, vectorSize=vectorSize, outputSize=outputSize)

    print("Model Created.")
    sess = tf.InteractiveSession(graph=nnModel.paperGraph)
    tf.global_variables_initializer().run()
    sess.run(tf.tables_initializer())

    if should_load:
        model_path = model_path
        tf.train.Saver().restore(sess, model_path)
        print("Model loaded.")
        
    with sess.as_default():
        accList = [] 
        trainModel(sess, embedModel, nnModel, iterations, trainData, trainTarget, testData, testTarget, configs, accList)
        
    return sess, nnModel

"""
We get layers from output to input so that we can backpropagate.
Then we calculate word importances for each word in input.
In the current model there is only one conv-pool layer so the layer_count is 1.
But in the medspecsearch models have 3 layers, so this model is different. 
We will use this model for LRP purposes.
"""

# Get weights, biases and activations to use in lrp method
def get_weights_biases_acts(layer_count, nnModel):
    """
    Return required weights, biases and activations for LRP.
    
    Parameters
    ---------
    layer_count : int
        Specifies how many conv-pool layer pairs in the model.
    nnModel : Models.CNN object
        NN model that we trained. Weights, biases activations of this model
        will be extracted and returned.
    Returns
    -------
    weights : list
        List containing weights in layers of model in reverse fashion.
    biases : list
        List containing biases in layers of model in reverse fashion.
    activations : list
        List containing activation layers in model in reverse fashion.
    """
    
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*kernel.*')
    biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*bias.*')

    activations = []
    if layer_count == 1:
        activations = [nnModel.cnnInput, nnModel.conv1, nnModel.blockPool, nnModel.h_pool_flat, nnModel.fc1, nnModel.scores]

    elif layer_count == 3:   
        activations = [nnModel.cnnInput, nnModel.conv1, nnModel.blockPool, nnModel.conv2, nnModel.blockPool2, nnModel.conv3,
                 nnModel.blockPool3, nnModel.h_pool_flat, nnModel.fc1, nnModel.scores]

    weights.reverse()
    biases.reverse()
    activations.reverse()
    
    return weights, biases, activations

def get_prediction_single_input(result):
    """
    This method returns prediction in human readable form.

    Parameters
    ----------
    result : dict
        Dictionary containing results returned from model after evaluation.

    Returns
    -------
    string
        Prediction in string form.
    """

    return reverse_class_dict[result["Prediction"][0]]