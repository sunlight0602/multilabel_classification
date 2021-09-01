# import matplotlib

import jsonlines
import numpy as np
import matplotlib.pyplot as plt

dataset = []
with jsonlines.open('./data/dataset_training.jsonl') as reader:
    for obj in reader:
        dataset.append(obj)
dataset = dataset[1:]

y_labels = [0, 0, 0, 0, 0, 0, 0, 0]
for data in dataset:
    label = (data[2], data[3], data[4])

    if label == (1, 0, 0):
        y_labels[0] += 1
    elif label == (0, 1, 0):
        y_labels[1] += 1
    elif label == (0, 0, 1):
        y_labels[2] += 1
    elif label == (1, 1, 0):
        y_labels[3] += 1
    elif label == (1, 0, 1):
        y_labels[4] += 1
    elif label == (0, 1, 1):
        y_labels[5] += 1
    elif label == (1, 1, 1):
        y_labels[6] += 1
    elif label == (0, 0, 0):
        y_labels[7] += 1

x_labels = ['(1,0,0)', '(0,1,0)', '(0,0,1)', '(1,1,0)', '(1,0,1)', '(0,1,1)', '(1,1,1)', '(0,0,0)']
x = np.arange(len(x_labels))
plt.bar(x, y_labels)
plt.xticks(x, x_labels)
plt.xlabel('(Spot, Restaurant, Lodging)')
plt.ylabel('Frequency')
plt.title('Frequency Observation')

for i in range(len(x_labels)):
    plt.text(i, y_labels[i], str(int((y_labels[i] / sum(y_labels) * 100)))+'%')

plt.savefig('frequency_observation.png', dpi=400)
# plt.show()
