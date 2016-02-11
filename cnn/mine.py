from nltk.tokenize import word_tokenize
import string, pandas as pd

# Set of punctuation marks to remove
punct = set(string.punctuation)

# Return true if token does not contain any character from bad_set
def clean(token, bad_set):
    return all([b not in token for b in bad_set])



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

print(tdm)
