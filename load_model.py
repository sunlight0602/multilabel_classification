import torch
# import numpy as np
# import pandas as pd 
import jsonlines

from preprocess import preprocessing_for_bert
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from predict import bert_predict

BATCH_SIZE = 16
TEST_DATA = './data/dataset_testing.jsonl'
OUTPUT_DATA = './data/prediction_testing.txt'
CONFIDENCE_THRESHOLD = 0.5
# TEST = './data/test.csv'
# test_csv = pd.read_csv(TEST)

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
model = torch.load('./saved_model/chi_model.pt', map_location=device)

# Read testing data
dataset = []
with jsonlines.open(TEST_DATA, 'r') as reader:
    for obj in reader:
        dataset.append(obj)
dataset = dataset[1:]

X = []
for data in dataset:
    X.append(data[0] + ' ' + data[1])

# Preprocess testing data
test_inputs, test_masks = preprocessing_for_bert(X)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

# Predict
# probs = bert_predict(model, test_dataloader, device)
all_logits = bert_predict(model, test_dataloader, device)

# Get predictions from the probabilities
preds = []
for idx, logit in enumerate(all_logits):
    pred = [1 if l>=CONFIDENCE_THRESHOLD else 0 for l in logit]
    preds.append([dataset[idx][0]] + pred)

# Write predictions
with open(OUTPUT_DATA, 'w') as f:
    for pred in preds:
        f.write(pred[0])

        if pred[1] == 1:
            f.write(' | Spot')
        if pred[2] == 1:
            f.write(' | Restaurant')
        if pred[3] == 1:
            f.write(' | Lodging')

        f.write('\n')
