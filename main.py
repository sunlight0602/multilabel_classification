import numpy as np
import pandas as pd 
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from preprocess import text_preprocessing, preprocessing_for_bert, detokenizer
from train import train
from model import BertClassifier, initialize_model
from utils import set_seed
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

# CONFIG
# TRAIN = './data/train.csv'
# TEST = './data/test.csv'
# TEST_LABEL = './data/test_labels.csv'
# SAMPLE = './data/sample_submission.csv'
MAX_LEN = 400
BATCH_SIZE = 16
EPOCHS = 5

# READ FILES
# train_csv = pd.read_csv(TRAIN)
# test_csv = pd.read_csv(TEST)
# test_label = pd.read_csv(TEST_LABEL)
# sample_sub = pd.read_csv(SAMPLE)
import jsonlines

dataset = []
with jsonlines.open('./data/dataset_training.jsonl') as reader:
    for obj in reader:
        dataset.append(obj)
dataset = dataset[1:]

X = []
Y = []
for data in dataset:
    X.append(data[0] + ' ' + data[1])
    Y.append(data[2:])

# SPLIT DATA
from sklearn.model_selection import train_test_split

# X = train_csv.comment_text.values
# Y = train_csv.toxic.values
# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.1, random_state = 666)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=666)


# PREPROCESS
print('Tokenizing data...')
train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_val)
# detokenizer(train_inputs[0])

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# BATCHING
# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)  # 相當於 zip()
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

# TRAIN MODEL
set_seed(42)    # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, device, epochs = EPOCHS)
model = train(bert_classifier, train_dataloader, device, optimizer, scheduler, val_dataloader, epochs=EPOCHS, evaluation=True)

# SAVE MODEL
# print('Save model')
# torch.save(model, './saved_model/chi_model.pt')


# =========================

# # TEST MODEL
# test_inputs, test_masks = preprocessing_for_bert(test_csv.comment_text)

# # Create the DataLoader for our test set
# test_dataset = TensorDataset(test_inputs, test_masks)
# test_sampler = SequentialSampler(test_dataset)
# test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = BATCH_SIZE)

# # PREDICT
# probs = bert_predict(bert_classifier, test_dataloader, device)

# # Get predictions from the probabilities
# threshold = 0.9
# preds = np.where(probs[:, 1] > threshold, 1, 0)

# # Number of tweets predicted toxic
# print("Number of tweets predicted toxic: ", preds.sum())

# test_csv['toxic'] = preds

# test_csv.drop('comment_text', axis = 1, inplace = True)

# test_csv['severe_toxic'] = 0
# test_csv['obscene'] = 0
# test_csv['threat'] = 0
# test_csv['insult'] = 0
# test_csv['identify_hate'] = 0

# test_csv.to_csv('./data/submission.csv')

