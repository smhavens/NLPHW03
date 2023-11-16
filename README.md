# NLPHW03

## Setup
### Requirements
- Download word_embeddings.py and optionally the models included in the repo to save time generating them.
- To use the classifier function, please run
  ```
  wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  tar -xzf aclImdb_v1.tar.gz
  ```
  to download the required imdb sentiment database.
- Run
  ```
  pip install wefe
  pip install datasets
  pip install apache_beam
  ```
  To use the Bias and Embeddings function. If you get an error about dill needing to be upgraded to install apache_beam, please run
  ```
  pip install dill --upgrade
  ```
- The program also uses the urban dictionary model and wiki-news model, but as the file sizes are too large to upload to github it is recommended that users download the prebuilt models here. If you want to load them yourself then go to these sites to get the [wiki-news-300d-1M-subwords.vec](https://fasttext.cc/docs/en/english-vectors.html) and [ud_basic.vec](https://smash.inf.ed.ac.uk/ud-embeddings/)
  
### How to Run
This program offers a handful of options to optimize your use. The options include:
- (--skip/-s) to skip the model training portion for the wikipedia models wiki2vec and skip2vec. (Requires all models to already be downloaded)
- (--extra/-e) to skip the wikipedia model generation but to go through the ud_basic and wiki-news vector loading
- (--bias/-b) to run the embedding bias function
- (--compare/-c) to run the embedding comparison function
- (--text/-t) to run the text classification using embeddings function
## Report

### Embeddings
For my simple wikipedia embeddings, I made use of the gensim embedding's option for bag-of-words(word2vec) or skip-gram(skip2vec). The training of these embeddings took a very long time, which I was happy gensim made easy to avoid with a save and load feature for their models. Between the BOW and Skip-gram models, both took roughly the same time to complete but in regards to comparisons the BOW had much better results both in regards to the delivered score (usually being larger then Skip-gram) and the word choices feeling more natural in my opinion. (As for the search 'witch - woman + man' BOW returned 'ghost' and 'demon' which are reasonable compared to Skip-gram's 'skillit' and 'mossflower'.) Though in their later use for bias analysis, in both categories Skip-gram reported a lower bias than BOW.

#### Comparisons
These were the results of my comparison of embeddings using the gensim BOW, Skip-gram, and the two pre-trained models on urban dictionary and wiki-news.

![embedding_comps](https://github.com/smhavens/NLPHW03/assets/55886989/e4df6dd4-8a5e-4d76-9628-940dc58be1ec)

I tried of variety of comparisons, from simple ones like 'dog', to more complex comparisons as seen with 'Pizza - Pepperoni + Pretzel'. I found the fasttext pre-trained wiki-news model to be the least interesting, as it often returned the keyword with added symbols (as you see with it returning 'death-' and 'death--' for most similar to 'death').

### Bias

![image](https://github.com/smhavens/NLPHW03/assets/55886989/70a233a6-6f93-4a49-9a4b-5c5244ae3674)

### Classification
![image](https://github.com/smhavens/NLPHW03/assets/55886989/9d27902e-e8dc-45f7-84c7-a2899c286743)

### Reflection
