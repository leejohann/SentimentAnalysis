from pathlib import Path

# get project root
current_dir = Path(__file__).resolve()
root = current_dir.parent.parent

#### DATA ####
data_path = root/'data'

# raw data
train_raw_path = data_path/'raw'/'twitter_training.csv'
test_raw_path = data_path/'raw'/'twitter_validation.csv'

# processed data
train_processed_path = data_path/'processed'/'processed_training.csv'
test_processed_path = data_path/'processed'/'processed_testing.csv'

# glove embeddings
glove_embeddings = data_path/'glove.6B.100d.txt'