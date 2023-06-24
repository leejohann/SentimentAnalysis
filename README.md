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
       
2. Model Building
   
    2.1 Embedding layer:
   
    * Embeddings are used to reduce high-dimensional feature spaces to denser feature spaces with low dimensions.
    * Embeddings are used in NLP, Recommendation Systems etc.
    * Keras uses methods like backpropagation and gradient descent update initialised matrices with the desired number of dimensions by minimising the specified loss function (in our case `categorical_accuracy_crossentropy`
    * For the basic model, we train the embedding layer using our own vocabulary, reducing vocabulary size of 17k to 100 dimensions in our embedding layer.
    * If we use a pre-trained embedding layer (like Word2Vec), we can choose if we want to keep the embeddings the same throughout the epochs, or fine-tune them over the epochs.
  
    2.2 Flatten layer: to flatten our embeddings to size 4300 (tweet size of 43 with 100 dimensions for each word embedding) for each tweet

   2.3 Dense layer: This layer takes in our flattened embeddings, applies weights/biases to our embeddings and applies the softmax funcion to obtain our probability distribution across the four classes we have. Then, the class with the highest distribution is our predicted class. During each epoch, the weights applied in this layer are updated to minimise our specified loss function.
   $$softmax(x_i) = \frac{\exp(x_i)}{\displaystyle\sum^{n}_{i=1} \exp(x_i)}$$

4. Model Fitting: With our specified Keras model, we fit the model using our training data
5. Model Prediction: With our trained model, we can predict on our test set to obtain predicted values
6. Evaluation and AccuracyL compare our actual and predicted values to obtain an accuracy and loss score
