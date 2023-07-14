import pandas as pd
import numpy as np
import project

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Embedding
from keras.initializers import Constant
from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import *

"""
Model 2 - Neural network with layers:
1. Pretrained embedding layer (GloVe)
2. Flatten
3. Dense with softmax activation
"""

class Model2_Creator:

    def __init__(self, vocab_size, indexed_vocab, max_tweet_size, 
                 train_indexed_dict, test_indexed_dict):
        self.vocab_size = vocab_size
        self.indexed_vocab = indexed_vocab
        self.max_tweet_size = max_tweet_size
        self.train_indexed_dict = train_indexed_dict
        self.test_indexed_dict = test_indexed_dict

        # create Keras model
        self.model = Sequential()


    def load_glove(self):
        glove_embeddings = {}
        with open(project.glove_embeddings) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                glove_embeddings[word] = coefs
        
        self.glove_embeddings = glove_embeddings


    def prepare_embeddings(self):
        # create embedding matrix for our vocabulary
        embedding_matrix = np.zeros((self.vocab_size + 2, 100))
        hits = 0
        misses = 0

        # store glove embeddings according to our index
        for word, word_index in self.indexed_vocab.items():
            word_embedding = self.glove_embeddings.get(word)
            if word_embedding is not None:
                embedding_matrix[word_index] = word_embedding
                hits += 1
            else:
                misses += 1

        self.hits = hits
        self.misses = misses
        self.embedding_matrix = embedding_matrix


    def build_model(self, is_trainable):
        # add layers to model
        self.model.add(Embedding(input_dim = self.vocab_size + 2,
                     output_dim = 100,
                     input_length = self.max_tweet_size,
                     embeddings_initializer=Constant(self.embedding_matrix),
                     trainable=is_trainable))
        self.model.add(Flatten())
        self.model.add(Dense(4, activation='softmax'))

        # compile the model
        self.model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['categorical_accuracy'])

        # summarize the model
        print(self.model.summary())

    
    def fit_model(self):
        # reshape and get training data
        train_x_df = pd.DataFrame.from_dict(self.train_indexed_dict, orient='index')
        train_df = pd.read_csv(project.train_processed_path, header=0, index_col=0)
        train_y_df = pd.get_dummies(train_df.sentiment)

        # Training the model
        self.model.fit(x=train_x_df, y=train_y_df, epochs=50, verbose=1)

    
    def model_predict(self):
        # reshape testing data
        test_x_df = pd.DataFrame.from_dict(self.test_indexed_dict, orient='index')
        test_df = pd.read_csv(project.test_processed_path, header=0, index_col=0)
        test_y_df = pd.get_dummies(test_df.sentiment)

        # Prediction and evaluation
        loss, accuracy = self.model.evaluate(test_x_df, test_y_df, verbose=1)

        return(loss, accuracy)