import pandas as pd
import re
import pickle
import sys


if (len(sys.argv) != 2):
    print("Usage: python3 " + sys.argv[0] + " \"[text]\"")
    sys.exit(1)

def remove_special_chars(str):
    return ''.join(e for e in str if e.isalnum())

# Load model.
word_counts_by_rating = dict()
total_word_count_by_rating = dict()

with open('word_count_by_rating.pkl', 'rb') as f:
    word_counts_by_rating = pickle.load(f)

with open("total_word_count_by_rating.pkl", "rb") as f:
    total_word_count_by_rating = pickle.load(f)

# Categories of ratings, from 1 to 5 stars.
categories = [ 1, 2, 3, 4, 5 ]

# Stopwords are ignored words.
stopwords = []
with open('stop_words.txt') as stopwords_file:
    stopwords = stopwords_file.readlines()

text = sys.argv[1]

# Inference.
word_dist = dict()
probs_per_category = dict()

for i in categories:
    probs_per_category[i] = 1

    for word in text.split(" "):

        stripped_word = remove_special_chars(word).lower()

        if stripped_word == "":
            continue

        # Keep track of distribution per word for pretty output at the end.
        if (word_dist.get(stripped_word, -1) == -1):
            word_dist[stripped_word] = dict()

        prob = (word_counts_by_rating[i][stripped_word] / total_word_count_by_rating[i])
        probs_per_category[i] *= prob

        # Keep track of distribution per word for pretty output at the end.
        word_dist[stripped_word][i] = prob

# Normalize
sum_probs_per_category = sum(probs_per_category.values())
for i in categories:
    probs_per_category[i] /= sum_probs_per_category

# Pretty output
print("\nInput text: ")
print(text)
print("\nProbability by category, as found by naive bayes classification: ")
print(probs_per_category)
print("\nInferred category: " + str(max(probs_per_category.keys(), key=lambda x: probs_per_category[x])))

# Show probability distribution for each word.
print("\n\nProbability distribution by word:\n")
for word in text.split(" "):
    stripped_word = remove_special_chars(word).lower()
    print(stripped_word + ":\t", end="")
    total = sum(word_dist[stripped_word].values())
    for i in categories:
        print(word_dist[stripped_word][i] / total, end="\t")
    print("\n")