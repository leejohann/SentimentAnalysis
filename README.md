# SentimentAnalysis
Sentiment classfication

Data retrieved from: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

# Objective

This project is a learning project on Natural Language Processing (NLP) and Deep Learning. 

The goal is to use a Neural Network for the embedding layer on processed text data, before fitting a classfication model to classify tweets into four categories: `Positive`, `Negative`, `Neutral`, `Irrelevant`.

# Outline
1. Data Processing
    The following steps were used to preprocess the tweets:
    1. Lowercasing
    2. Remove all punctuation except for apostrophes
    3. Whitespace tokenisation
    4. Removing stopwords
    5. Removing words with numbers (numbers, usernames etc.)
    6. POS tagging
    7. Lemmatisation
2. Embedding layer
    * Embeddings are used to reduce high-dimensional feature spaces to denser feature spaces with low dimensions.
    * Embeddings are used in NLP, Recommendation Systems etc.
    * To obtain the embedding matrix, methods like backpropagation and gradient descent update initialised matrices with the desired number of dimensions by minimising a specified loss function

3. Fitting a classification model
4. Evaluation