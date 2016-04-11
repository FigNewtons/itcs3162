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


def kmeans(df, d, sections):
    nrow = df.shape[0]
    init = [ np.random.choice(k, 1)[0] for k in sections.values() ]
    K = len(sections)

    centroids = df.iloc[init]
    centroids.index = range(0, K)

    def assign_row(row):
        row_label = centroids.apply(lambda c: d(row, c), axis = 1).idxmin()
        return centroids.index.get_loc(row_label)

    for i in range(10):
        labels = df.apply(assign_row, axis = 1)
        centroids = df.groupby(labels).mean()

    sse = 0
    for i in range(nrow):
        sse += d(df.iloc[i], centroids.iloc[labels[i]])**2

    return (labels, centroids, sse)

def accuracy(predict, actual):
    return sum(predict == actual) / len(actual)


def run():
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


    true_labels = [0] * len(articles)
    for i, key in enumerate(sections.keys()):
        for val in sections[key]:
            true_labels[val] = i

    clean_articles = list(map(clean, articles))

    counts = {}
    for idx, article in enumerate(clean_articles):
        for word in article:
            if word not in counts:
                counts[word] = [0] * len(clean_articles)
            
            counts[word][idx] += 1

    tdm = pd.DataFrame(counts)
    tdm.index = titles
    # tdm.to_csv("tdm.csv")

    elabels, ecentroids, esse = kmeans(tdm, dist, sections)
    clabels, ccentroids, csse = kmeans(tdm, cosine, sections)
    jlabels, jcentroids, jsse = kmeans(tdm > 0, jaccard, sections)

    with open("results.txt", "w") as f:
        print("Euclidean: Accuracy = {0}, SSE = {1:.4f}".format(accuracy(elabels, true_labels), esse), file = f)
        print("Cosine: Accuracy = {0}, SSE = {1:.4f}".format(accuracy(clabels, true_labels), csse), file = f)
        print("Jaccard: Accuracy = {0}, SSE = {1:.4f}".format(accuracy(jlabels, true_labels), jsse), file = f)

if __name__ == '__main__':
    run()


