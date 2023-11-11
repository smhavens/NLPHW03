from datasets import load_dataset
import shutil
import json
from collections import defaultdict
import multiprocessing
import gensim
from gensim import corpora
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import fasttext
from gensim.test.utils import datapath
from wefe.datasets import load_bingliu
from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.utils import plot_queries_results, run_queries
import pandas as pd
import gensim.downloader as api

from wefe.metrics import WEAT
from wefe.datasets import load_weat
from wefe.utils import run_queries
from wefe.utils import plot_queries_results
import math
import sys
import glob
import os
import argparse
import nltk
import scipy.sparse
import numpy as np
import string
import io


'''STEPS FOR CODE:
1. Train word embeddings on Simple English Wikipedia;
2. Compare these to other pre-trained embeddings;
3. Quantify biases that exist in these word embeddings;
4. Use your word embeddings as features in a simple text classifier;
'''


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    print("Hello", n, d)
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        # print(data)
        
    print(data)
    return data


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


def compare_embeddings(cbow, skip, urban, fasttext):
    '''COMPARE EMBEDDINGS'''
    print("Most Similar to dog")
    print("cbow", cbow.wv.most_similar(positive=['dog'], negative=[], topn=2))
    print("skip", skip.wv.most_similar(positive=['dog'], negative=[], topn=2))
    print("urban", urban.most_similar(positive=['dog'], negative=[], topn=2))
    print("fasttext", fasttext.most_similar(positive=['dog'], negative=[], topn=2))
    
    print("\nMost Similar to Pizza - Pepperoni + Pretzel")
    print("cbow", cbow.wv.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("skip", skip.wv.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("urban", urban.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['pizza', 'pretzel'], negative=['pepperoni'], topn=2))
    
    print("\nMost Similar to witch - woman + man")
    print("cbow", cbow.wv.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("skip", skip.wv.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("urban", urban.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['witch', 'man'], negative=['woman'], topn=2))
    
    print("\nMost Similar to mayor - town + country")
    print("cbow", cbow.wv.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("skip", skip.wv.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("urban", urban.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    print("fasttext", fasttext.most_similar(positive=['mayor', 'country'], negative=['town'], topn=2))
    
    print("\nMost Similar to death")
    print("cbow", cbow.wv.most_similar(positive=['death'], negative=[], topn=2))
    print("skip", skip.wv.most_similar(positive=['death'], negative=[], topn=2))
    print("urban", urban.most_similar(positive=['death'], negative=[], topn=2))
    print("fasttext", fasttext.most_similar(positive=['death'], negative=[], topn=2))


def quantify_bias(cbow, skip, urban, fasttext):
    '''QUANTIFY BIASES'''
    '''Using WEFE, RNSB'''
    
    RNSB_words = [
        ['christianity'],
        ['catholicism'],
        ['islam'],
        ['judaism'],
        ['hinduism'],
        ['buddhism'],
        ['mormonism'],
        ['scientology'],
        ['taoism']]
    
    weat_wordset = load_weat()
    
    models = [WordEmbeddingModel(cbow.wv, "CBOW"),
              WordEmbeddingModel(skip.wv, "skip-gram"),
              WordEmbeddingModel(urban, "urban dictionary"),
              WordEmbeddingModel(fasttext, "fasttext")]
    
    # Define the 10 Queries:
    print(weat_wordset["science"])
    queries = [
        # Flowers vs Insects wrt Pleasant (5) and Unpleasant (5)
        Query([['christianity', 'catholicism', 'islam', 'judaism', 'hinduism', 'buddhism', 'mormonism', 'scientology', 'taoism'], weat_wordset['science']],
            [weat_wordset['pleasant_5'], weat_wordset['unpleasant_5']],
            ['Religion', 'Science'], ['Pleasant(5)', 'Unpleasant(5)']),
        
        Query([['christianity', 'catholicism', 'islam', 'judaism', 'hinduism', 'buddhism', 'mormonism', 'scientology', 'taoism'], weat_wordset['science']],
            [weat_wordset['male_terms'], weat_wordset['female_terms']],
            ['Religion', 'Science'], ['Male terms', 'Female terms']),

        # # Instruments vs Weapons wrt Pleasant (5) and Unpleasant (5)
        # Query([weat_wordset['instruments'], weat_wordset['weapons']],
        #     [weat_wordset['pleasant_5'], weat_wordset['unpleasant_5']],
        #     ['Instruments', 'Weapons'], ['Pleasant(5)', 'Unpleasant(5)']),

        # # European american names(5) vs African american names(5)
        # # wrt Pleasant (5) and Unpleasant (5)
        # Query([
        #     weat_wordset['european_american_names_5'],
        #     weat_wordset['african_american_names_5']
        # ], [weat_wordset['pleasant_5'], weat_wordset['unpleasant_5']],
        #     ['European american names(5)', 'African american names(5)'],
        #     ['Pleasant(5)', 'Unpleasant(5)']),

        # # European american names(7) vs African american names(7)
        # # wrt Pleasant (5) and Unpleasant (5)
        # Query([
        #     weat_wordset['european_american_names_7'],
        #     weat_wordset['african_american_names_7']
        # ], [weat_wordset['pleasant_5'], weat_wordset['unpleasant_5']],
        #     ['European american names(7)', 'African american names(7)'],
        #     ['Pleasant(5)', 'Unpleasant(5)']),

        # # European american names(7) vs African american names(7)
        # # wrt Pleasant (9) and Unpleasant (9)
        # Query([
        #     weat_wordset['european_american_names_7'],
        #     weat_wordset['african_american_names_7']
        # ], [weat_wordset['pleasant_9'], weat_wordset['unpleasant_9']],
        #     ['European american names(7)', 'African american names(7)'],
        #     ['Pleasant(9)', 'Unpleasant(9)']),

        # # Male and female names wrt Career and family
        # Query([weat_wordset['male_names'], weat_wordset['female_names']],
        #     [weat_wordset['career'], weat_wordset['family']],
        #     ['Male names', 'Female names'], ['Career', 'Family']),

        # # Math and arts wrt male and female terms
        # Query([weat_wordset['math'], weat_wordset['arts']],
        #     [weat_wordset['male_terms'], weat_wordset['female_terms']],
        #     ['Math', 'Arts'], ['Male terms', 'Female terms']),

        # # Science and arts wrt male and female terms
        # Query([weat_wordset['science'], weat_wordset['arts_2']],
        #     [weat_wordset['male_terms'], weat_wordset['female_terms']],
        #     ['Science', 'Arts 2'], ['Male terms', 'Female terms']),

        # # Mental and Physical disease wrt Temporary and Permanent
        # Query([weat_wordset['mental_disease'], weat_wordset['physical_disease']],
        #     [weat_wordset['temporary'], weat_wordset['permanent']],
        #     ['Mental disease', 'Physical disease'], ['Temporary', 'Permanent']),

        # # Young people names and Old people names disease wrt Pleasant(9) and Unpleasant(9)
        # Query(
        #     [weat_wordset['young_people_names'], weat_wordset['old_people_names']],
        #     [weat_wordset['pleasant_9'], weat_wordset['unpleasant_9']],
        #     ['Young peoples names', 'Old peoples names'],
        #     ['Pleasant(9)', 'Unpleasant(9)'])
    ]
    
    # wefe_results = run_queries(WEAT,
    #                   queries,
    #                   models,
    #                   metric_params={
    #                       'secondary_preprocessor_args': {
    #                           'lowercase': True
    #                       },
    #                       'warn_not_found_words': True
    #                   }).T.round(2)

    wefe_results = run_queries(WEAT,
                                queries,
                                models,
                                metric_params ={
                                    'preprocessors': [
                                        {},
                                        {'lowercase': True }
                                    ]
                                },
                                warn_not_found_words = True
                                ).T.round(2)
    # joined_results = pd.concat([wefe_results, original_results], axis=1)
    # joined_results = joined_results.iloc[:, [2,0,3,1]] # reorder the columns
    # joined_results
    
    wefe_results
    plot_queries_results(wefe_results).show()


    # RNSB_words = [
    #     ["swedish"],
    #     ["irish"],
    #     ["mexican"],
    #     ["chinese"],
    #     ["filipino"],
    #     ["german"],
    #     ["english"],
    #     ["french"],
    #     ["norwegian"],
    #     ["american"],
    #     ["indian"],
    #     ["dutch"],
    #     ["russian"],
    #     ["scottish"],
    #     ["italian"],
    # ]
    
    # bing_liu = load_bingliu()
    # # Create the query
    # query = Query(RNSB_words,
    #             [bing_liu['positive_words'], bing_liu['negative_words']])
    # # Fetch the models

    # # glove = (api.load('glove-wiki-gigaword-300'),
    # #                         'glove-wiki-gigaword-300')

    # cbow_WEM = WordEmbeddingModel(cbow.wv, "CBOW")
    # # Run the queries
    # cbow_results = RNSB().run_query(query, cbow_WEM)
    # # conceptnet_results = RNSB().run_query(query, conceptnet)
    # # Show the results obtained with glove
    # cbow_fig = px.bar(
    #     pd.DataFrame(cbow_results['negative_sentiment_distribution'],
    #                 columns=['Word', 'Sentiment distribution']), x='Word',
    #     y='Sentiment distribution', title='Glove negative sentiment distribution')
    # cbow_fig.update_yaxes(range=[0, 0.2])
    # cbow_fig.show()
    
    # model = WordEmbeddingModel(
    #     cbow.wv
    # )
    
    # model = WordEmbeddingModel(api.load('glove-wiki-gigaword-300'), 'Glove wiki')
    
    # print(type(RNSB_words))
    
    # query = Query(RNSB_words, [bing_liu["positive_words"], bing_liu["negative_words"]])
    
    # evaluate(query, 'fasttext-wiki-news-subwords-300', 'FastText')


def evaluate(
    query: Query, gensim_model_name, short_model_name: str, model_args: dict = {}
):
    # Fetch the model
    model = WordEmbeddingModel(
        api.load(gensim_model_name), short_model_name, **model_args
    )

    # Run the queries
    results = RNSB().run_query(
        query, model, holdout=True, print_model_evaluation=True, n_iterations=100
    )

    # Show the results obtained with glove
    fig = px.bar(
        pd.DataFrame(
            results["negative_sentiment_distribution"].items(),
            columns=["Word", "Sentiment distribution"],
        ),
        x="Word",
        y="Sentiment distribution",
        title=f"{short_model_name} Negative Sentiment Distribution",
    )

    fig.update_yaxes(range=[0, 0.2])
    fig.show()


def text_classifier(cbow):
    '''SIMPLE TEXT CLASSIFIER'''
    

def main():
    parser = argparse.ArgumentParser(
        prog='word_embedding',
        description='This program will train a word embedding model using simple wikipedia.',
        epilog='To skip training the model and to used the saved model "word2vec.model", use the command --skip or -s.'
    )
    parser.add_argument('-s', '--skip', action='store_true')
    parser.add_argument('-e', '--extra', action='store_true')
    parser.add_argument('-b', '--bias', action='store_true')
    parser.add_argument('-c', '--compare', action='store_true')
    parser.add_argument('-t', '--text', action='store_true')
    
    args = parser.parse_args()
    skip_model = None
    cbow_model = None
    ud_model = None
    wiki_model = None
    if args.compare:
        if args.skip:
            print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            ud_model = KeyedVectors.load("urban2vec.model")
            wiki_model = KeyedVectors.load("wiki2vec.model")
        elif args.extra:
            print("Extra mode")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False) 
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False) 
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False)
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False)
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        compare_embeddings(cbow_model, skip_model, ud_model, wiki_model)
    elif args.bias:
        if args.skip:
            print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            ud_model = KeyedVectors.load("urban2vec.model")
            wiki_model = KeyedVectors.load("wiki2vec.model")
        elif args.extra:
            print("Extra mode")
            cbow_model = Word2Vec.load("word2vec.model")
            skip_model = Word2Vec.load("skip2vec.model")
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False) 
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False) 
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
            wiki_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subwords.vec", binary=False)
            ud_model = KeyedVectors.load_word2vec_format("ud_basic.vec", binary=False)
            wiki_model.save("wiki2vec.model")
            ud_model.save("urban2vec.model")
        quantify_bias(cbow_model, skip_model, ud_model, wiki_model)
    elif args.text:
        if args.skip:
            print("Skipping")
            cbow_model = Word2Vec.load("word2vec.model")
        else:
            cbow_model, skip_model = train_embeddings()
        text_classifier(cbow_model)
        
    print("No errors?")
    

if __name__ == "__main__":
    main()