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

# getting stopwords
stopwords = stopwords.words('english')

# POS tagging and lemmatisation
#nltk.download('averaged_perceptron_tagger')  # Download POS tagger data
#nltk.download('wordnet')

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