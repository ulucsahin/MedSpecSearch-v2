# MedSpecSearch-v2

Download required data-embeddings-models from 
https://drive.google.com/drive/folders/1RHj-AnlXyEIKRGa7WmNg1eE_8ezsGnvl?usp=sharing.

Some Embeddings (specified in txt file in zip that you will download from the link) needed to be downloaded seperately, such as GoogleNews Word2Vec Embeddings.

Tutorial Jupyter Notebooks are explained below.

All methods are documented.


# Scripts
Below are list of ipynb tutorial files that show how to use scripts. 

### iCliniq Data Scraping.ipynb 
This notebook shows how to use icliniq_data_scraper module to collect data from icliniq website. Downloaded data will be saved to hard drive.

Shows ssage of: <br>
- icliniq_data_scraper.py
<br>
<br>

### Embedding Training.ipynb
This notebook shows how to train and save Word2Vec and FastText embeddings.

Uses: <br>
- Gensim <br>
- Pandas <br>

### Training and Saving Model.ipynb
This notebook shows how to train and save a tensoflow model into hard drive.

Uses: <br>
- helper.py<br>
- EmbedHelper.py<br>
- Models.py<br>

### Restoring Model.ipynb
This notebook shows how to restore a pre-trained model from hard drive.

Uses:<br>
- helper.py<br>
- EmbedHelper.py<br>
- Models.py<br>

### Getting Predictions.ipynb
This notebook shows how to get predictions from a trained model.

Uses: <br>
- helper.py<br>
- EmbedHelper.py<br>
- DataLoader.py<br>
- Models.py<br>

### Hospital Data.ipynb
Shows how to use hospitals.py script to get hospital results for specified medical specialty and region.

Shows usage of: <br>
- hospitals.py

### Phrases.ipynb
Shows how to use newsgroups or iCliniq data (named fold0 in script) to find frequently used phrases in data. Also shows training and saving embeddings of Phrase-replaced data.

Shows usage of: <br>
- tfidf_mesh.py

### Preprocessing Input and Text Translation.ipynb
This notebook shows usage of Text Preprocessing and Text translation (From any language to English, input language is automatically deduced, Google doesnt ask extra money for automatic language detection)

Uses: <br>
- DataLoader.py

Shows usage of: <br>
- DataHandler


An authentication token for Google Translation API is required to use Turkish translation in this project. Steps below explains how to get a token and use it. 

## How to get aut.json file for Google API
- Login to https://console.developers.google.com <br>
- Click "Credentials" on the left side under "APIs & Services". <br>
- If there is no previously created project available, create a new project. <br>
- Click "Create Credentials" and select "Service account key". <br>
- Select Service Account, choose JSON format as key type and click Create. <br>
- Rename downloaded file as "aut.json" and put it under data folder of this project.


## Config File properties <br>
```
configs = {
    "vectorSize":300,
    "trainNewModel":True,
    "dataColumn":"question",
    "maxLength":128,
    "batchSize":64,
    "embeddingType":embedDict[2],
    "PreEmbed":True,
    "restore":True,
    "model_type":"CNN_3Layer" # Options are : "CNN" (1 layer) , "CNN_3Layer", "RNN_LSTM"
}
```
- vectorSize: Dimension size of embedding vectors. We have used for our embeddings. <br>
- trainNewModel: Specifies if a new model should be trained or not. <br>
- dataColumn: Specifies which column should be used as data in csv files. Can be different for each csv file. <br>
- maxLength: Maximum sentence length in words. Data instances with more than maxLength will be cut to 128 words. <br>
- batchSize: Batch size during training. <br>
- embeddingType: Specifies which embedding should be used. <br>
- PreEmbed: Specifies if loaded embeddings should be used or not. Should be "True" in most cases. <br>
- restore: Specifies if model should be restored <br>
<br>
Some of these specifications are not currently used but kept for backwards compatability.



