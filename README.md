# MedSpecSearch-v2

TODO:
Add auth.json file tutorial.


Download required data-embeddings-models from 
https://drive.google.com/drive/folders/1RHj-AnlXyEIKRGa7WmNg1eE_8ezsGnvl?usp=sharing.

Some Embeddings (specified in txt file in zip that you will download from the link) needed to be downloaded seperately, such as GoogleNews Word2Vec Embeddings.

Example Usage Jupyter Notebook explains important parts of the system.

All methods are documented.

Below are list of ipynb tutorial files:
TODO: Split Example Usage and list here

### Hospital Data.ipynb
Shows how to use hospitals.py script to get hospital results for specified medical specialty and region.

uses: hospitals.py

### Phrases.ipynb
Shows how to use newsgroups or iCliniq data (named fold0 in script) to find frequently used phrases in data. Also shows training and saving embeddings of Phrase-replaced data.

Uses: tfidf_mesh.py

<!--- ### TF-IDF and MESH.ipynb
Shows how to use tfidf_mesh module to --->

### iCliniq Data Scraping.ipynb
Downloads data from https://www.icliniq.com. Downloaded data will be saved to hard drive.

