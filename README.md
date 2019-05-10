# MedSpecSearch-v2

# Scripts
Below are list of ipynb tutorial files that show how to use scripts. 

### iCliniq Data Scraping.ipynb 
Downloads data from https://www.icliniq.com. Downloaded data will be saved to hard drive.

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




An authentication token for Google Translation API is required to use Turkish translation in this project. Steps below explains how to get a token and use it. 

## How to get aut.json file for Google API
- Login to https://console.developers.google.com <br>
- Click "Credentials" on the left side under "APIs & Services". <br>
- If there is no previously created project available, create a new project. <br>
- Click "Create Credentials" and select "Service account key". <br>
- Select Service Account, choose JSON format as key type and click Create. <br>
- Rename downloaded file as "aut.json" and put it under data folder of this project.



Download required data-embeddings-models from 
https://drive.google.com/drive/folders/1RHj-AnlXyEIKRGa7WmNg1eE_8ezsGnvl?usp=sharing.

Some Embeddings (specified in txt file in zip that you will download from the link) needed to be downloaded seperately, such as GoogleNews Word2Vec Embeddings.

Tutorial Jupyter Notebooks are explained below.

All methods are documented.



