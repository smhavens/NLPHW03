from datasets import load_dataset
import shutil
import json
from collections import defaultdict
import multiprocessing
import gensim
import gensim.downloader as api
from gensim import corpora
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import math
import sys
import glob
import os
import argparse
import nltk
import scipy.sparse
import numpy as np
import string


'''STEPS FOR CODE:
1. Train word embeddings on Simple English Wikipedia;
2. Compare these to other pre-trained embeddings;
3. Quantify biases that exist in these word embeddings;
4. Use your word embeddings as features in a simple text classifier;
'''

def train_embeddings():
    '''TRAIN WORD EMBEDDINGS
    This will be making use of the dataset from wikipedia and the first step'''
    dataset = load_dataset("wikipedia", "20220301.simple")
    cores = multiprocessing.cpu_count()
    # check the first example of the training portion of the dataset :
    # print(dataset['train'][0])
    dataset_size = len(dataset)
    
    ### BUILD VOCAB ###
    # print(type(dataset["train"][0]))
    vocab = set()
    vocab_size = 0
    count = 0
    ## Generate vocab and split sentances and words?
    data = []
    for index, page in enumerate(dataset["train"]):
        document = page["text"]
        document = document.replace("\n", ". ")
        # print(document)
        for sent in document.split("."):
            # print("Sentance:", sent)
            new_sent = []
            clean_sent =[s for s in sent if s.isalnum() or s.isspace()]
            clean_sent = "".join(clean_sent)
            for word in clean_sent.split(" "):
                if len(word) > 0:
                    new_word = word.lower()
                    # print("Word:", new_word)
                    if new_word[0] not in string.punctuation:
                        new_sent.append(new_word)
            if len(new_sent) > 0:
                data.append(new_sent)
                # print("New Sent:", new_sent)
    
    
    for index, page in enumerate(dataset["train"]):
        # print(page["text"])
        # for text in page:
        #     print(text)
        text = page["text"]
        clean_text = [s for s in text if s.isalnum() or s.isspace()]
        clean_text = "".join(clean_text)
        clean_text = clean_text.replace("\n", " ")
        # text = text.replace('; ', ' ').replace(", ", " ").replace("\n", " ").replace(":", " ").replace(". ", " ").replace("! ", " ").replace("? ", " ").replace()
        
        for word in clean_text.split(" "):
            # print(word)
            if word != "\n" and word != " " and word not in vocab:
                vocab.add(word)
                vocab_size += 1
            # if index == 10:
            #     break
            # print(f"word #{index}/{count} is {word}")
        count += 1
            
    print(f"There are {vocab_size} vocab words")
    
    embeddings_model = Word2Vec(
                     data,
                     epochs= 10,
                     window=10,
                     vector_size= 50)
    embeddings_model.save("word2vec.model")
    
    skip_model = Word2Vec(
                     data,
                     epochs= 10,
                     window=10,
                     vector_size= 50,
                     sg=1)
    skip_model.save("skip2vec.model")
    
    embeddings_model = Word2Vec.load("word2vec.model")
    skip_model = Word2Vec.load("skip2vec.model")
    # embeddings_model.train(dataset, total_examples=dataset_size, epochs=15)
    # print(embeddings_model['train'])
    # print(embeddings_model.wv["france"])
    return embeddings_model, skip_model


def compare_embeddings(cbow, skip):
    '''COMPARE EMBEDDINGS'''


def quantify_bias(cbow, skip):
    '''QUANTIFY BIASES'''


def text_classifier(cbow, skip):
    '''SIMPLE TEXT CLASSIFIER'''
    

def main():
    parser = argparse.ArgumentParser(
        prog='word_embedding',
        description='This program will train a word embedding model using simple wikipedia.',
        epilog='To skip training the model and to used the saved model "word2vec.model", use the command --skip or -s.'
    )
    parser.add_argument('-s', '--skip', action='store_true')
    
    args = parser.parse_args()
    skip_model = None
    cbow_model = None
    if args.skip:
        cbow_model = Word2Vec.load("word2vec.model")
        skip_model = Word2Vec.load("skip2vec.model")
    else:
        cbow_model, skip_model = train_embeddings()
    compare_embeddings(cbow_model, skip_model)
    quantify_bias(cbow_model, skip_model)
    text_classifier(cbow_model, skip_model)
    print("No errors?")
    

if __name__ == "__main__":
    main()