import torch
import numpy as np
import pandas as pd 
from preprocess import preprocessing_for_bert
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from predict import bert_predict


# SETTING UP THE GPU IF POSSIBLE
# if torch.cuda.is_available():       
#     device = torch.device("cuda")
#     print(f'There are {torch.cuda.device_count()} GPU(s) available.')
#     print('Device name:', torch.cuda.get_device_name(0))
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")
device = torch.device("cpu")

# LOAD MODEL
model = torch.load('./saved_model/save.pt', map_location=device)

# model.eval()

# CONFIG and READ FILE
BATCH_SIZE = 16
# TEST = './data/new_input.csv'
TEST = './data/test.csv'
test_csv = pd.read_csv(TEST)

# TEST MODEL
test_inputs, test_masks = preprocessing_for_bert(test_csv.comment_text)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

# PREDICT
probs = bert_predict(model, test_dataloader, device)

# Get predictions from the probabilities
threshold = 0.9
preds = np.where(probs[:, 1] > threshold, 1, 0)
print(preds)

# Number of tweets predicted toxic
print("Number of inputs predicted toxic: ", preds.sum())

test_csv['toxic'] = preds

test_csv.drop('comment_text', axis = 1, inplace = True)

test_csv['severe_toxic'] = 0
test_csv['obscene'] = 0
test_csv['threat'] = 0
test_csv['insult'] = 0
test_csv['identify_hate'] = 0

test_csv.to_csv('./data/submission.csv')
