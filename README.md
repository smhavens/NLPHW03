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
For my Bias Study extension, I looked into WEAT and created my own subcategory focused on religion, as it was never mentioned in their paper or documentation.
![image](https://github.com/smhavens/NLPHW03/assets/55886989/70a233a6-6f93-4a49-9a4b-5c5244ae3674)

As seen with Religion and Art wrt Career and Family & Religion and Weapons wrt Male and Female Terms, there is bias to be acknowledged, especially in regards to the BOW Simple Wikipedia model and to a similar extent the Skip-gram Simple Wikipedia model. I believe these embeddings could be useful in understanding how religion is deeply ingrained in certain aspects of life, such as it appears to be with art here.

### Classification
For classification, I used the Simple Wikipedia BOW model to see its impact on classifying IMDB sentiment scores. I based the code mainly off of what was covered in lecture 11, replacing the vector with a new d-dimensional vector holding the averaged embeddings of the IMDB database documents.
![image](https://github.com/smhavens/NLPHW03/assets/55886989/9d27902e-e8dc-45f7-84c7-a2899c286743)

Compare these results with the original one using TF*IDF:

![image](https://github.com/smhavens/NLPHW03/assets/55886989/d42a58cc-81ee-4f21-965b-4a7f8499a3ed)

The results using embeddings instead seem to have some improvements (0's precision & 1's recall) but generally have worse scores overall. Potentially this embedding improvement would work better with a larger training base for the model (and not just simple wikipedia) or to use a database more likely to use terminology applicable to sentiment analysis and opinion pieces.

### Reflection
This assignment gave me a much better understanding of both the application of embeddings but also how classifiers are developed. It was also interesting to see just how different results can be with different training datasets (urban dictionary vs fasttext vs simple wikipedia) and also how it is best to train on applicable datasets for what you will later test on (as I found with the sentiment classifier using simple wikipedia embeddings).

I faced challenges with many of the models having deprecated features and needing to learn new variations to have everything returning expected results, especially with Gensim's updates to the KeyedVector model. There was also wrapping my head around exactly how to use word embeddings, as even though we go over them in class, applying them myself meant I really needed to look over what I'd learned.
