from datasets import load_dataset
import shutil
import json
from collections import defaultdict
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


'''STEPS FOR CODE:
1. Train word embeddings on Simple English Wikipedia;
2. Compare these to other pre-trained embeddings;
3. Quantify biases that exist in these word embeddings;
4. Use your word embeddings as features in a simple text classifier;
'''

def train_embeddings():
    '''TRAIN WORD EMBEDDINGS
    This will be making use of the dataset from wikipedia and the first step'''
    dataset = load_dataset("wikipedia", "20220301. simple")
    # check the first example of the training portion of the dataset :
    # print(dataset['train'][0])
    dataset_size = len(dataset)
    embeddings_model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    embeddings_model.save("word2vec.model")
    
    embeddings_model = Word2Vec.load("word2vec.model")
    embeddings_model.train(dataset, total_examples=dataset_size)


def compare_embeddings(model):
    '''COMPARE EMBEDDINGS'''


def quantify_bias(model):
    '''QUANTIFY BIASES'''


def text_classifier(model):
    '''SIMPLE TEXT CLASSIFIER'''
    

def main():
    model = train_embeddings()
    compare_embeddings(model)
    quantify_bias(model)
    text_classifier(model)
    

if __name__ == "__main__":
    main()