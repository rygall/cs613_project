import numpy as np
import sys
import csv
import functions


# SET K VALUE HERE
k = 30

# DATA PREPROCESSING
# opening and reading data from data file
data = []
with open('CTG.csv') as ctg:
    ctg_read = csv.reader(ctg, delimiter=',')
    data = np.array(list(ctg_read))
    data = data[2:]
    data = np.array(data, dtype=float)

# random shuffle data
np.random.seed(0)
np.random.shuffle(data)

# splitting training and validation data
split_index = int((2/3) * np.size(data, axis=0))
training_data = data[:split_index, :-2]
training_classes = data[:split_index, -1]
validation_data = data[split_index:, :-2]
validation_classes = data[split_index:, -1]

# z-scoring training  data
mean = np.mean(training_data, axis=0)
std = np.std(training_data, axis=0, ddof=1)
training_data = (training_data - mean) / std
validation_data = (validation_data - mean) / std

# get validation vector
validation_vector = functions.myKNN(training_data, training_classes, validation_data, k)

# determine accuracy
values = np.unique(validation_classes)
confusion_matrix = np.zeros((len(values), len(values)), dtype=int)
for i in range(len(validation_classes)):
    confusion_matrix[int(validation_vector[i])-2][int(validation_classes[i])-2] +=1  

print('=========CONFUSION MATRIX=========')
print(confusion_matrix)
print('==================================')

total = 0
for i in range(0, np.size(confusion_matrix, axis=0)):
    array = confusion_matrix[i]
    total += array[i]
accuracy = (total) / np.size(validation_vector, axis=0)
print('Validation Accuracy:', accuracy)