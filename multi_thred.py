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

    for dir, destination in zip(dirs, destinations):
        for file in os.listdir(dir):
        #     breaking = True
        #     break
        # if breaking:
        #     break

            # print(f"{file.replace('.parquet', '')}_stemmed_no_rt.parquet")
            if (f"{file.replace('.parquet', '')}_processed.parquet" not in os.listdir(dir)) and ('processed' not in file):
                print(file)
                t1 = time.time()

                # df = pd.read_csv(dir + '/' + file, encoding='utf-8')
                # pool = multiprocessing.Pool(16)
                # L = [pool.map(stem, df['tweet'].to_list())]
                # df['stemmed_tweet'] = L[0]
                # df.to_csv(dir + '/' + file, index=False)

                # # open csv
                table = pq.read_table(os.path.join(dir, file))
                # to pandas
                df = pd.DataFrame(table.to_pydict())
                # pick first 100 rows
                # df = df[:100]
                # drop na from text
                # df = df.dropna(subset=['tweet'])
                # sample 100
                # delete duplicates
                # df = df.drop_duplicates(subset=['id'])

                RT = True
                if RT:
                    texts = df.text.to_list()
                    new_texts = []
                    rt = []
                    for text in texts:
                        if text[:2] == 'RT':
                            new_texts.append(' '.join(text.split(': ')[1:]))
                            rt.append(True)
                        else:
                            new_texts.append(text)
                            rt.append(False)
                    texts_dates = [(str(idx), str(text), str(date)[:10]) for (idx, text, date) in
                                   zip(df['id'], new_texts, df['created'])]
                else:
                    texts_dates = [(str(idx), str(text), str(date)[:10]) for (idx, text, date) in zip(df['id'], df['text'], df['created']) if text[:2] != 'RT']
                    new_texts = [text for (idx, text, date) in texts_dates]
                del table
                print(len(texts_dates))
                print('loaded Tweets')

                texts_set = list(set(new_texts))
                print(len(texts_set))

                pool = multiprocessing.Pool(16)
                L = [pool.map(stem, texts_set)]
                print('stemmed')

                dictionary = dict(zip(texts_set, [line for line in L[0]]))
                # dates = [line[2] for line in texts_dates]
                print('dictionary created')

                # ids = [line[0] for line in texts_dates]
                texts = [dictionary[line[1]] for line in texts_dates]
                pool.close()
                del pool
                # save
                # pydict = {'id': ids, 'text': texts, 'date': dates, 'rt': rt}
                df['stemmed'] = texts
                table = pa.Table.from_pandas(df)
                # overwrite
                pq.write_table(table, os.path.join(dir, file.replace('.parquet', '_processed.parquet')))

                # pq.write_table(table, os.path.join(destination, f"{file.replace('.parquet', '')}_stemmed_rt.parquet"))

                t2 = time.time()
                # count minutes
                print((t2 - t1) / 60)


