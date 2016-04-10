from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

import string, numpy as np, pandas as pd

import os
from os.path import split
from glob import glob

# ----- Similarity measures and helper functions -----

def jaccard(v1, v2):
    Mand = sum(v1 & v2)
    Mor = sum(v1 | v2)
    return Mand / Mor

def dist(v1, v2):
    return sum(np.square(v1 - v2))**0.5

def norm(v):
    return dist(v, np.zeros(v.shape))

def cosine(v1, v2):
    return sum(v1 * v2) / (norm(v1) * norm(v2))

def euclidean(v1, v2):
    return 1 / (1 + dist(v1, v2))

# ------------ NLP functions and whatnot ------------

porter = PorterStemmer()

stop_words = set(stopwords.words('english'))
stop_words.update(list(string.punctuation))

def clean(article):
    return [porter.stem(w.lower()) for w in wordpunct_tokenize(article) if w.lower() not in stop_words]


if __name__ == '__main__':

    article_list = glob(os.path.join("articles", "*", "*"))
    articles = []
    sections = {}
    titles = []

    for idx, article in enumerate(article_list):
        titles.append(split(article)[1])
        section = split(split(article)[0])[1]
        
        if section in sections:
            sections[section].append(idx)
        else:
            sections[section] = [idx]

        with open(article, "r") as f:
            articles.append(''.join(f.readlines()))

    K = len(sections)

    clean_articles = list(map(clean, articles))

    counts = {}
    for idx, article in enumerate(clean_articles):
        for word in article:
            if word not in counts:
                counts[word] = [0] * len(clean_articles)
            
            counts[word][idx] += 1

    tdm = pd.DataFrame(counts)
    tdm.index = titles
    print(tdm.head())

