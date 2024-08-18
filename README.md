# Review-Classifier
Naive Bayes Classifier for Amazon Review Data.

## Datasets
This program doesn't ship with, but uses, the datasets from [https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

Really, what matters is that your dataset is in the 'jsonl' format, and contains a list of objects with a 'rating' and 'text' field.

## How-To
### venv
Start by first, optionally, creating a virtual environment.

```
$ python3 -m venv env
```

Then, activate the virtual environment

```
$ source env/bin/activate
```

Now, pip install the requirements

```
$ pip3 install -r requirements.txt
```

### Build Model
Build your model. This step will take a few minutes.

```
$ python3 build_model.py [dataset.jsonl]
```

### Inference

Input a test phrase into the second command line argument for classifier.py to be classified based on a 1-5 star rating. For optimal results, use a test review that could plausibly be within the category of your dataset. For example, if you choose the "Appliances" dataset, use a sample review that could be for a toaster, oven, or etc.

Now, run inference on your model by using the classifier.

```
$ python3 classifier.py "[text]"
```