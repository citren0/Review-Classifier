import pandas as pd
import re
import pickle 
import sys


if (len(sys.argv) != 2):
    print("Usage: python3 " + sys.argv[0] + " [dataset.jsonl]")
    sys.exit(1)


def remove_special_chars(str):
    return "".join(e for e in str if e.isalnum())


categories = [ 1, 2, 3, 4, 5 ]

# Read JSONL file into pandas dataframe.
dataset = pd.read_json(sys.argv[1], lines=True)

# Stopwords are ignored words.
stopwords = []
with open("stop_words.txt") as stopwords_file:
    stopwords = stopwords_file.readlines()

# Transform data into a useful format.
reviews_by_rating = dict()
for i in categories:
    reviews_by_rating[i] = dataset[dataset["rating"] == i]["text"].values

# Build model.
word_counts_by_rating = dict()
total_word_count_by_rating = dict()
for i in categories:
    word_counts_by_rating[i] = dict()
    total_word_count_by_rating[i] = 0
    
    # Populate dictionary.
    for review in reviews_by_rating[i]:
        for word in review.split(" "):

            stripped_word = remove_special_chars(word).lower()

            if word in stopwords:
                continue

            if stripped_word == "":
                continue

            total_word_count_by_rating[i] += 1

            if word_counts_by_rating[i].get(stripped_word, -1) == -1:
                word_counts_by_rating[i][stripped_word] = 0
            else:
                word_counts_by_rating[i][stripped_word] += 1


with open("word_count_by_rating.pkl", "wb") as f:
    pickle.dump(word_counts_by_rating, f)

with open("total_word_count_by_rating.pkl", "wb") as f:
    pickle.dump(total_word_count_by_rating, f)