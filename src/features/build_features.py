import pandas as pd
import nltk
import re
import tensorflow as tf
import torch
import torchtext
import random

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences

"""
This file transform the processed data into tokens to feed into our model
"""

"""
Getting stuff
"""
# getting stopwords
stopwords = stopwords.words('english')

# POS tagging and lemmatisation
#nltk.download('averaged_perceptron_tagger')  # Download POS tagger data
#nltk.download('wordnet')

"""
Preprocessing and tokenising text
"""
# function to convert POS tags to wordnet tags
def convert_pos_tag(tag):
    """ 
    Takes in POS tags to POS tags that are used by wordnet (for lemmatisation)
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
    
# function to pre-process text df
def preprocess_text(processed_data_path, remove_apostrophes):
    """
    Takes in df of tweets in str form, indexed by tweet_id, and returns dictionary of tweets processed:
        1. lowercasing
        2. remove punctuation except for apostrophes
        3. whitespace tokenisation
        4. remove stopwords
        5. remove numbers and words with numbers
        6. POS tagging
        7. lemmatisation

    :param tweet_df: df of tweets, indexed by tweet_id, tweet_content is a string
    :param stopword_list: preloaded list of stopwords
    :param remove_apostrophes: bool value indicating if we want to remove apostrophes
    """
    # load processed data
    tweet_df = pd.read_csv(processed_data_path, header=0, index_col=0)
    
    # dictionary to store all tokens
    tweet_dict = {tweet_id: None for tweet_id in tweet_df.index}
    
    for tweet_id in tweet_dict.keys():
        # extract text
        tweet = tweet_df.loc[tweet_id]['tweet_content']
        
        # lowercasing
        tweet = tweet.lower()

        # remove all punctuation 
        if remove_apostrophes:
            tweet = re.sub(r'[^\w\d\s]+', ' ', tweet)
        else:
            tweet = re.sub(r'[^\w\d\s\']+', ' ', tweet)
        
        # whitespace tokenisation
        tweet_tokens = nltk.WhitespaceTokenizer().tokenize(tweet)
        
        # removing stopwords
        tweet_tokens = [word for word in tweet_tokens if word not in stopwords]

        # remove any words with numbers in them
        tweet_tokens = [token for token in tweet_tokens if not re.match(r'.*\d.*', token)]

        # POS tagging and convert to wordnet tags
        tagged_tweet_tokens = nltk.pos_tag(tweet_tokens)

        # Lemmatisation
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for token, pos_tag in tagged_tweet_tokens:
            wn_pos = convert_pos_tag(pos_tag)
            if wn_pos is not None:
                lemma = lemmatizer.lemmatize(token, pos=wn_pos)
            else:
                lemma = lemmatizer.lemmatize(token)
            
            lemmas.append(lemma)

        # update tweet dictionary
        tweet_dict[tweet_id] = lemmas

    return tweet_dict

"""
Some other helper functions
"""
# function to find vocabulary, vocabulary size and max tweet size (train_df + test_df)
def find_vocab(tweet_dict_list):
    """ 
    Takes in a list of tweet dictionaries to find total vocabulary size 
    and max tweet size (for padding)
    """
    # store
    vocabulary = set()
    max_tweet_size = 0

    for dict in tweet_dict_list:
        for tokens in dict.values():
            # vocabulary
            vocabulary.update(tokens)

            # finding max tweet size for padding
            if len(tokens) > max_tweet_size:
                max_tweet_size = len(tokens)

    vocab_size = len(vocabulary)
    
    return vocabulary, vocab_size, max_tweet_size


# function to pad and index our dictionaries of tweets
def index_tweets(tweet_dict_list):
    """
    Takes in list of tweet dictionaries to index each dictionary based on aggregated 
        vocabulary
    """
    # extract vocabulary
    vocabulary, vocab_size, max_tweet_size = find_vocab(tweet_dict_list)

    # Index vocabulary
    vocab_index = list(range(2, vocab_size + 2))
    indexed_vocab = {word:index for word, index in zip(vocabulary, vocab_index)}

    # index each dictionary
    indexed_dict_list = []
    for tweet_dict in tweet_dict_list:

        indexed_dict = {}

        for tweet_id, tweet in tweet_dict.items():
        # storing index of every word in each tweet
            tweet_indexed = []
            for word in tweet:
                word_index = indexed_vocab[word]
                tweet_indexed.append(word_index)
            # store tweet indexes
            indexed_dict[tweet_id] = tweet_indexed

            # pad tweet here
            pad_amount = max_tweet_size - len(tweet_indexed)
            if pad_amount > 0:
                tweet_indexed.extend([0]* pad_amount) # padding
        
        # append to list
        indexed_dict_list.append(indexed_dict)
        
    
    return indexed_dict_list, indexed_vocab