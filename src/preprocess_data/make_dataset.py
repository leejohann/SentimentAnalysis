import pandas as pd
import project

class DatasetCreator:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def load_data(self):
        # read and identify all columns
        self.data = pd.read_csv(self.raw_data_path, header=None,
                names=['tweet_id', 'entity', 'sentiment', 'tweet_content'])

    def preprocess(self):
        # idk what's going on but only take the first of each tweet_id
        self.data = self.data.groupby('tweet_id', as_index = True).first()

    def save_data(self):
        self.data.to_csv(self.processed_data_path)

    def make_dataset(self):
        self.load_data()
        self.preprocess()
        self.save_data()



