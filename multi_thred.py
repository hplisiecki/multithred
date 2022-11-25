import os
import pandas as pd
from bs4 import BeautifulSoup
from pyMorfologik import Morfologik
from pyMorfologik.parsing import ListParser
import pickle
from tqdm import tqdm
import os
import nltk
import requests
# tokenizer polish
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os
import nltk
import pandas as pd
import numpy as np
import time

with open('data/lemmatizer_dictionary.pickle', 'rb') as handle:
    lema_dict = pickle.load(handle)

dir = 'https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt'
# download the file
r = requests.get(dir)
with open('data/stopwords.txt', 'wb') as handle:
    handle.write(r.content)
# read the file
with open('data/stopwords.txt', 'r') as f:
    stopwords = f.read().splitlines()

# stopwords.extend(['http', '@'])

parser = ListParser()
stemmer = Morfologik()
import string
def stem(sentence):
    # remove interpunction
    if sentence[:2] == 'RT':
        sentence = sentence[2:]
    sentence = "".join([ch for ch in sentence if ch not in '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'])
    words = str(sentence).split(' ')
    words = [word for word in words if word not in stopwords and '@' not in word and 'http' not in word]
    # if len(words) == 0:
    #     print('empty')
    tweet = ' '.join(words)
    morf = stemmer.stem([tweet.lower()], parser)
    string = ''
    for i in morf:
        if i[0] in lema_dict.keys():
            string += lema_dict[i[0]] + ' '
        else:
            try:
                string += list(i[1].keys())[0] + ' '
            except:
                string += i[0] + ' '
    string = string[:-1]

    return string

if __name__ == '__main__':


    dirs = [r'Z:\Data\Twitter\tweets_from_users\tweets_arrow', r'Z:\Data\Twitter\tweets_from_users\onuce_tweets_arrow']
    destinations = [r'Z:\Data\Twitter\A\stems_rt', r'Z:\Data\Twitter\E\stems_rt', r'Z:\Data\Twitter\B1\stems_rt', r'Z:\Data\Twitter\B2\stems_rt']
    breaking = False


    texts_set = ['ale dzisiaj zjadłem obiad' , 'zjadłem obiad dzisia bardzo']
    pool = multiprocessing.Pool(16)
    L = [pool.map(stem, texts_set)]

    pool.close()
    del pool



