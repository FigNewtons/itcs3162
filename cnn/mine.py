from nltk.tokenize import word_tokenize
import string, numpy as np, pandas as pd

# Set of punctuation marks to remove
punct = set(string.punctuation)

# Return true if token does not contain any character from bad_set
def clean(token, bad_set):
    return all([b not in token for b in bad_set])


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


# Return pairwise similarity matrix for given term-doc matrix and sim function
def compute_similarity(tdm, sim):
    N = tdm.shape[0]
    matrix = np.zeros((N,N))

    for i in range(0, N):
        for j in range(i, N):
            matrix[i, j] = sim(tdm.loc[i], tdm.loc[j])

    return pd.DataFrame(matrix)



if __name__ == '__main__':

    with open("stories.txt", "r") as f:
        stories = f.readlines()

    clean_stories = [[ word.lower() for word in word_tokenize(story) if clean(word, punct) ] for story in stories ]

    # Unfold clean_stories for distinct words
    word_bank = set([word for story in clean_stories for word in story])


    # Create term-document matrix (TDM) as a dictionary
    counts = dict((word, [0] * len(clean_stories)) for word in word_bank )

    for index, story in enumerate(clean_stories):
        for word in story:
            counts[word][index] += 1


    tdm = pd.DataFrame(counts)

    jaccard_df = compute_similarity(tdm.apply(lambda x: x > 0), jaccard)
    cosine_df = compute_similarity(tdm, cosine)
    euclidean_df = compute_similarity(tdm, euclidean)


    print("\nJaccard\n")
    print(jaccard_df)

    print("\nCosine\n")
    print(cosine_df)

    print("\nEuclidean\n")
    print(euclidean_df)











