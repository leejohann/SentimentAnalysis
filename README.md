# SentimentAnalysis
Sentiment classfication

Data retrieved from: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

# Objective

This is a learning project on Natural Language Processing (NLP) and Deep Learning. 

The goal is to use a Neural Network so we can add an embedding layer on processed text data, then fit a classfication model to classify tweets into four categories: `Positive`, `Negative`, `Neutral`, `Irrelevant`.

# Outline
1. Data Processing
    The following steps were used to preprocess each tweet:
    1. Lowercasing
    2. Remove all punctuation except for apostrophes
    3. Whitespace tokenisation
    4. Removing stopwords
    5. Removing words with numbers (numbers, usernames etc.)
    6. POS (Part-of-speech) tagging
    7. Lemmatisation
2. Model building
    Embedding layer:
        * Embeddings are used to reduce high-dimensional feature spaces to denser feature spaces with low dimensions.
        * Embeddings are used in NLP, Recommendation Systems etc.
        * To obtain the embedding matrix, methods like backpropagation and gradient descent update initialised matrices with the desired number of dimensions by minimising a specified loss function

        For the basic model, we reduce our vocabulary size of 16.6k to 100 dimensions in our embedding layer.

3. Fitting a classification model
4. Evaluation and Accuracy