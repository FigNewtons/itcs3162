from nltk.tokenize import word_tokenize
import string, numpy as np, pandas as pd
import os


# ----- Similarity measures and helper functions ------

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

# -----------------------------------------------------

def clean(token, bad_set):
    """ Return true if token doesn't contain any character from bad_set. """
    return all([b not in token for b in bad_set])

def compute_similarity_to_df(tdm, sim):
    """Return pairwise similarity matrix.
    
    tdm is a term-document matrix
    sim is a similarity function
    
    """
    N = tdm.shape[0]
    matrix = np.zeros((N,N))

    for i in range(0, N):
        for j in range(i, N):
            matrix[i, j] = sim(tdm.loc[i], tdm.loc[j])
            matrix[j, i] = matrix[i, j]

    return pd.DataFrame(matrix)

def rank(sim_matrix):
    """Return a sorted rank matrix using similarity values.

    Row indices represent document
    Column indices represent ranking (lowest to highest) 
    Value represents the compared document    
    
    """
    N = sim_matrix.shape[0]
    rank_df = sim_matrix.apply(np.argsort, axis = 1)
    rank_df.columns = list(range(N, 0, -1))
    
    return rank_df

def compute_similarity_to_dict(tdm, sim):
    """Return a dictionary of similarity measures and pairs"""
    N = tdm.shape[0]
    matrix = np.zeros((N,N))

    sim_dict = {}
    for i in range(0, N):
        for j in range(i, N):
            s = round(sim(tdm.loc[i], tdm.loc[j]), 6)

            if s not in sim_dict:
                sim_dict[s] = [(i, j)]
            else:
                sim_dict[s].append((i,j))

    return sim_dict

def output(sim, info):
    """Output similarity ranks to file. """
    sim_dict = compute_similarity_to_dict(info["tdm"], info["fun"])

    with open("output/" + sim + ".txt", "w") as f:
        print(sim + " similarity ranks\nMeasure   Pair(s)", file = f)
        for key in sorted(sim_dict, reverse = True):
            print("{0:.6f}: {1}".format(key, sim_dict[key]), file = f)

#-----------------------------------------------------

if __name__ == '__main__':

    with open("stories.txt", "r") as f:
        stories = f.readlines()

    # Set of punctuation marks to remove
    punct = set(string.punctuation)

    clean_stories = [[ word.lower() for word in word_tokenize(story) if clean(word, punct) ] for story in stories ]

    # Unfold clean_stories for distinct words
    word_bank = set([word for story in clean_stories for word in story])

    # Create term-document matrix (TDM) as a dictionary
    counts = dict((word, [0] * len(clean_stories)) for word in word_bank )

    for index, story in enumerate(clean_stories):
        for word in story:
            counts[word][index] += 1

    if not os.path.exists("output"):
        os.mkdir("output")

    tdm = pd.DataFrame(counts)
    tdm.to_csv("output/tdm.csv")

    similarities = { "jaccard"  : { "tdm": tdm.apply(lambda x: x > 0), "fun": jaccard },
                     "cosine"   : { "tdm": tdm, "fun": cosine },
                     "euclidean": { "tdm": tdm, "fun": euclidean } }

    for sim, info in similarities.items():
        output(sim, info)
    
