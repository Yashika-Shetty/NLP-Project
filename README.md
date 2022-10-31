<h1 align="center">Welcome to NLP Project </h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.3-blue.svg?cacheSeconds=2592000" />
  <img src="https://img.shields.io/badge/npm-%3E%3D5.5.0-blue.svg" />
  <img src="https://img.shields.io/badge/node-%3E%3D9.3.0-blue.svg" />
  <a href="#" target="_blank">
    <img alt="License: None" src="https://img.shields.io/badge/License-None-yellow.svg" />
  </a>
</p>

## Description
- This project uses Lstm and n gram model for text prediction in English and Hindi Language!

  ## Loading And Pre-processing the data
  ### Load the Data
    In this step, we upload the text file which will act as a corpus for our machine learning model. Depending on our input text file, the text prediction will be done. This text file will be saved as UTF-8 encoded text file. Then , Clean Text and remove numbers.
  ### Sentence & Word Tokenization
    The data extracted from the file is tokenized into sentences and words.We convert all tokens into lower case so that words which are capitalized (for example, at the start of a sentence) in the original text are treated the same as the lowercase versions of the words.Then we append each tokenized list of words into a list of tokenized sentences.
  ### Splitting the Dataset
    The dataset is divided into train and test with a 80-20 split.
    How many total characters do we have in our training text? It is given by ```chars = sorted(list(set(raw_text))) ``` (List of every character)
  ### Word Frequency
    We don't use all the tokens (words) appearing in the data for training. Instead,we used the more frequently used words.We count how many times each word appears in the data. Character sequences must be encoded as integers. Each unique character will be assigned an integer value. Create a dictionary of characters mapped to integer values. Do the reverse so we can print our predictions in characters and not integers. Then, summarize the data.
  ### Applying count threshold
    If the model is performing autocomplete, but encounters a word that it never saw during training, it won't have an input word to help it determine the next word to suggest. The model will not be able to predict the next word because there are no counts for the current word.This 'new' word is called an 'unknown word', or out of vocabulary (OOV) words.To handle unknown words during prediction, use a special token to represent all unknown words 'unk'.We then create a list of the most frequent words in the training set, called the closed vocabulary by converting all the other words that are not part of the closed vocabulary to the token 'unk'.The words that appear count_threshold times or more are in the closed vocabulary.We will then preprocess train and test data.
  ### Develop LSTM based language models
    Let’s now design our model. Because there is obviously going to be sequential, temporal structure underlying the training data, we will use an LSTM layer, a type of advanced recurrent neural network. In fact, this is all we need, unless we want to create a deep neural network spanning multiple layers. However, training such a model would cost a lot of time and computational resource. For the sake of simplicity, we will build a simple model with a single LSTM layer. The output layer is going to be a dense layer with vocab_size number of neurons, activated with a softmax function. We can thus interpret the index of the biggest value of the final array to correspond to the most likely character.


## Prerequisite

- Python 3.8.2 (recomended)
- Keras
- Tensorflow
## Install

```sh
git clone https://github.com/Yashika-Shetty/NLP-Project
pip install --upgrade pip
pip install tensorflow
```

## Usage

This project uses Lstm and n-gram model for text prediction in English and Hindi Language! You need to replace ```enghin.txt``` with the txt file you want to take as input.


## Run tests

The predefined value of Epoch is set to 50 , you can change it as per convenience/ requirement  .

## Explore it
Explore it and twist it to your own use case, in case of any question feel free to reach me out directly ```yashikashetty15@gmail.com```

## Issues
Incase you have any difficulties or issues while trying to run the script you can raise it on the issues.

## Pull Requests
If you have something to add I welcome pull requests on improvement , you're helpful contribution will be merged as soon as possible.

## Give it a Star
If you find this repo useful , give it a star so as many people can get to know it.

## Author

👤 **Yashika Shetty**

* Github: [@Yashika-Shetty](https://github.com/Yashika-Shetty)
* LinkedIn: [@Yashika-Shetty](https://linkedin.com/in/Yashika-Shetty)

👤 **Ameya Shetty**

* LinkedIn: [@Ameya-Shetty](https://in.linkedin.com/in/ameya-shetty-472b1b19b)

👤**Bansari Vora**

* LinkedIn: [@Bansari-Vora](https://in.linkedin.com/in/bansari-vora-04bba91aa)

👤**Stuti Trivedi**

* LinkedIn: [@Stuti-Trivedi](https://in.linkedin.com/in/stuti-trivedi-a9428a1b2)
