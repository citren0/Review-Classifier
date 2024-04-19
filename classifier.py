import pandas as pd
import re


def add_or_increment(dictionary, value):
    if value in dictionary:
        dictionary[value] += 1
    else:
        dictionary[value] = 1


def remove_keys_from_dict(dictionary, keys):
    for key in keys:
        dictionary.pop(key, "none")

    return dictionary


def clean_and_populate_dictionary(dictionary, text_df):
    # For each entry in the dataframe.
    for index, row in text_df.iterrows():

        # Split each word, making the naive bayes assumption.
        for word in row['text'].split():

            lower_word = word.lower()

            # Use regex to remove all emojies or special characters.
            stripped_word_no_emojis_or_symbols = re.sub(r'[^a-zA-Z0-9]', '', lower_word)

            # If there is still a word remaining, add or increment its count in the dictionary.
            if stripped_word_no_emojis_or_symbols != '':
                add_or_increment(dictionary, stripped_word_no_emojis_or_symbols)


def get_bayes_distribution(text, df, rows_by_rating, stop_words, unique_words_by_rating, unique_word_count_by_rating):
    # Take the naive bayes assumption for this text and assume that each word is independent of one another.

    # We have 5 classes, 1, 2, 3, 4, 5, for each class calculate the P(text | c) P(c) then max and that is your class.
    # I will use Laplace to not have to deal with 0 probabilities for words that are uncommon.

    # Store probability distribution by star class in this dictionary.
    classes = dict()

    # Initialize probabilities to 1.0
    for curr_class in range(1, 5 + 1):
        classes[curr_class] = 1.0

    for curr_class in classes:
        # Do P(c) term first.
        classes[curr_class] *= (len(rows_by_rating[curr_class]) / len(df))

        # Now do P(feature | class) terms.
        for word in text.split():
            lower_word = word.lower()

            # Ignore stop words.
            if lower_word not in stop_words:

                # Add laplace smoothing to each term to avoid 0 probabilities.
                classes[curr_class] *= (unique_words_by_rating[curr_class].get(lower_word, 0) + 1) / (unique_word_count_by_rating[curr_class] + total_set_size)


    # Normalize the distribution
    normalizing_factor = sum(list(classes.values()))
    if normalizing_factor != 0:
        for curr_class in classes:
            classes[curr_class] /= normalizing_factor
    
    return classes






# stop_words.txt contains common words that do not contribute to the sentiment of a sentence, ex. "I", "You", "is"
stop_words_file = open('stop_words.txt', 'r')
stop_words = stop_words_file.readlines()
stop_words_stripped = [word.replace("\n", "") for word in stop_words]


# This is our main dataset of reviews.
df = pd.read_json("./Appliances.jsonl", lines=True)


# Cut the dataframe up by star rating.
rows_by_rating = dict()
rows_by_rating[5] = df[df['rating'] == 5]
rows_by_rating[4] = df[df['rating'] == 4]
rows_by_rating[3] = df[df['rating'] == 3]
rows_by_rating[2] = df[df['rating'] == 2]
rows_by_rating[1] = df[df['rating'] == 1]


# Each star rating will have a dictionary in which the program will store the word counts.
unique_words_by_rating = dict()
unique_words_by_rating[5] = dict()
unique_words_by_rating[4] = dict()
unique_words_by_rating[3] = dict()
unique_words_by_rating[2] = dict()
unique_words_by_rating[1] = dict()

clean_and_populate_dictionary(unique_words_by_rating[5], rows_by_rating[5])
clean_and_populate_dictionary(unique_words_by_rating[4], rows_by_rating[4])
clean_and_populate_dictionary(unique_words_by_rating[3], rows_by_rating[3])
clean_and_populate_dictionary(unique_words_by_rating[2], rows_by_rating[2])
clean_and_populate_dictionary(unique_words_by_rating[1], rows_by_rating[1])


# Remove stop words from each star rating.
unique_words_by_rating[5] = remove_keys_from_dict(unique_words_by_rating[5], stop_words_stripped)
unique_words_by_rating[4] = remove_keys_from_dict(unique_words_by_rating[4], stop_words_stripped)
unique_words_by_rating[3] = remove_keys_from_dict(unique_words_by_rating[3], stop_words_stripped)
unique_words_by_rating[2] = remove_keys_from_dict(unique_words_by_rating[2], stop_words_stripped)
unique_words_by_rating[1] = remove_keys_from_dict(unique_words_by_rating[1], stop_words_stripped)


# What is the number of keys in each dictionary.
unique_word_count_by_rating = dict()
unique_word_count_by_rating[1] = len(unique_words_by_rating[1])
unique_word_count_by_rating[2] = len(unique_words_by_rating[2])
unique_word_count_by_rating[3] = len(unique_words_by_rating[3])
unique_word_count_by_rating[4] = len(unique_words_by_rating[4])
unique_word_count_by_rating[5] = len(unique_words_by_rating[5])


# Total number of unique words.
all_keys = []
for curr_class in unique_words_by_rating.keys():
    all_keys += list(unique_words_by_rating[curr_class].keys())
# Use set to prevent duplicates.
total_set_size = len(set(all_keys))



text = "This product was awful in every way. It refused to work, and then customer service got mad at me for breaking it!"
# Remove potential emojis or special characters from our sample text.
text_stripped = re.sub(r'[^a-zA-Z0-9\ ]', '', text)

print(text_stripped)
# Print the distribution of probabilities for each class under the naive bayes assumption.
print(get_bayes_distribution(text_stripped, df, rows_by_rating, stop_words_stripped, unique_words_by_rating, unique_word_count_by_rating))