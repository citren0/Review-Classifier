# Review-Classifier
Naive Bayes Classifier for Amazon Review Data.

This program doesn't ship with, but uses, the datasets from [https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

Input a test phrase into the 'text' field in classifier.py to be classified based on a 1-5 star rating. For optimal results, use a test review that could plausibly be within the category of your dataset. For example, if you choose the "Appliances" dataset, use a sample review that could be for a toaster, or oven.

A generic sample review exists already within classifier.py

## Run
```
$ python3 classifier.py
```