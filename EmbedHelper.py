import numpy as np
import gensim

class EmbeddingHandler:
    embedDict = {
        1:"Fast Text",
        2:"Google News",
        3:"HealthTap",
        4:"Pubmed",
        5:"Glove",
        6:"iCliniq Trigram",
        7:"iCliniq default"
        }

    def __init__(self,embedType,trainNewModel,vectorSize,embedPath):
        self.embedType = embedType
        self.embedPath = embedPath
        self.model = self.getEmbeddingModel(embedType,trainNewModel,vectorSize)
        if(self.model == None):
            raise Exception("Failed to create embedding model")

      
    @staticmethod
    def loadFastTextModel():
        """
        This method is not yet implemented.
        """

        raise Exception("Fast Text Not Supported Yet")

        # model = FastText.load("data/htFastText.embed")
        # model = model.wv
        return model

    def getEmbeddingModel(self,embeddingType,trainNewModel,vectorSize):
        """
        Loads and returns embedding model.


        Parameters
        ----------
        embeddingType : string
            Type of embedding to be loaded. Types are available 
            in embedDict in this class.
        trainNewModel : bool
            Specifies if this method should train a new model or not.
        vectorSize : int
            Vector size of embeddings to be loaded or trained.

        Returns
        -------
        Word2VecKeyedVectors
            Embedding model that is loaded or trained.
        """

        #Google News
        if(embeddingType == EmbeddingHandler.embedDict[2]):
            print("Loading Google News")
            model = gensim.models.KeyedVectors.load_word2vec_format(self.embedPath+"/GoogleNews-vectors-negative300.bin",
                                                                    binary=True)

            return model

        #Fast Text
        elif(embeddingType == EmbeddingHandler.embedDict[1]):
            print("Loading Fast Text")
            model = EmbeddingHandler.loadFastTextModel()
            return model

        elif(embeddingType == EmbeddingHandler.embedDict[3]):
            print("Loading HT Word2Vec")
            return gensim.models.KeyedVectors.load(self.embedPath+"/healthTapEmbedding.embed")

        elif (embeddingType == EmbeddingHandler.embedDict[4]):
            print("Loading Pubmed")
            return gensim.models.KeyedVectors.load_word2vec_format(self.embedPath + "/wikipedia-pubmed-and-PMC-w2v.bin",binary=True)

        elif(embeddingType == EmbeddingHandler.embedDict[5]):
            print("Loading Glove")
            return gensim.models.KeyedVectors.load_word2vec_format(self.embedPath+"/glove840kW2V.txt")

        elif(embeddingType == EmbeddingHandler.embedDict[6]):
            print("Loading iCliniq Trigram Embeds (W2V)")
            return gensim.models.KeyedVectors.load("Embeddings//icliniq_trigram//icliniq_trigram.w2v")

        elif(embeddingType == EmbeddingHandler.embedDict[7]):
            print("Loading iCliniq Default Embeds (W2V)")
            return gensim.models.KeyedVectors.load("Embeddings//icliniq_default//icliniq_default.w2v")
            
        else:
            print("Embedding Does not Exist")
        
    def vectorizeSentence(self, sentence):
        """
        Vectorized given sentence by using loaded or trained embeddings.

        Parameters
        ----------
        sentence : list
            list of strings(words)

        Returns
        -------
        ndarray
            Sentence in vectorized form by using embedding.
        """
        embeddedSentence = []
        vectorSize = self.model.vector_size

        for word in sentence:
            embedding = np.zeros(vectorSize)
            if(word == "[None]"):
                embedding = np.zeros(vectorSize)
            else:
                if(word in self.model):
                    embedding = self.model[word]
                    embedding=np.array(embedding)
                else:
                    embedding=np.zeros(vectorSize)

            embeddedSentence += [embedding]

        return embeddedSentence

    def vectorizeBatch(self,batchData):
        """
        Similar to vectorize sentence, but performs on whole batch.
        
        Parameters
        batchData : ndarray
            ndarray of sentences splitted to words.

        Returns
        -------
        ndarray
           Numpy array of vectorized batch.
        """
        embedList = []

        for sentence in batchData:
            embedList += [self.vectorizeSentence(sentence)]

        return np.array(embedList)

