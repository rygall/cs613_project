import numpy as np
import sys
import csv
import functions
#np.set_printoptions(threshold=sys.maxsize)


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

# converting all features into binary features
training_data_mean = np.mean(training_data, axis=0)
# training data conversion
for i in range(0, np.size(training_data, axis=0)):
    for j in range(0, np.size(training_data, axis=1)):
        if training_data[i][j] <= training_data_mean[j]:
            training_data[i][j] = 0
        else:
            training_data[i][j] = 1
# validation data conversion
for i in range(0, np.size(validation_data, axis=0)):
    for j in range(0, np.size(validation_data, axis=1)):
        if validation_data[i][j] <= training_data_mean[j]:
            validation_data[i][j] = 0
        else:
            validation_data[i][j] = 1

# ensuring the array data type is int
training_data = np.array(training_data, dtype=int)
training_classes = np.array(training_classes, dtype=int)
validation_data = np.array(validation_data, dtype=int)
validation_classes = np.array(validation_classes, dtype=int)

# obtaining decision tree
decisionTree = functions.myDT(training_data, training_classes)

# generating validation predictions
validation_predictions = []
for sample in validation_data:
    classification = decisionTree.classify(sample)
    validation_predictions.append(classification)

# determine accuracy
values = np.unique(validation_classes)
confusion_matrix = np.zeros((len(values), len(values)), dtype=int)
for i in range(0, len(validation_classes)):
    confusion_matrix[int(validation_predictions[i])-1][int(validation_classes[i])-1] +=1  

print('=========CONFUSION MATRIX=========')
print(confusion_matrix)
print('==================================')

total = 0
for i in range(0, np.size(confusion_matrix, axis=0)):
    array = confusion_matrix[i]
    total += array[i]
accuracy = (total) / np.size(validation_predictions, axis=0)
print('Validation Accuracy:', accuracy)