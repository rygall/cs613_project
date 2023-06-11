import numpy as np
import math
import preprocessing as data

# Linear Regression Function
def myLR(training_data, training_classes, validation_data):
    # PARAMETERS
    max_epochs = 50000
    learning_rate = 0.01

    # generate random weights
    np.random.seed(0)
    w = np.random.uniform(low=-0.0001, high=0.0001, size=np.size(training_data, axis=1))

    for epoch in range(0, max_epochs):

            # determine all y_hats based on training data
            y_hat = np.matmul(training_data, w)

            # compute log loss for training data and add to tracking array
             
            # calculate the gradient
            r2 = np.transpose(training_data)
            r3 = np.subtract(y_hat, training_classes)
            r4 = np.matmul(r2, r3)
            n = np.size(y_hat, axis=0)
            gradient = (1 / n) * r4
            
            # update weights based on average gradient
            w = np.subtract(w, (learning_rate*gradient))

    # determine all y_hats based on validation data
    validation_vector = np.matmul(validation_data, w)

    return validation_vector


# Data Preprocessing
# opening and reading data from data file
data = data.getContinousData()

# random shuffle data
np.random.seed(0)
np.random.shuffle(data)

# splitting training and validation data
split_index = int((2/3) * np.size(data, axis=0))
training_data = data[:split_index, :-1]
training_classes = data[:split_index, -1]
validation_data = data[split_index:, :-1]
validation_classes = data[split_index:, -1]

# insert bias feature
training_data = np.insert(training_data, 0, 1, axis=1)
validation_data = np.insert(validation_data, 0, 1, axis=1)

# get validation vector
validation_vector = myLR(training_data, training_classes, validation_data)

# RMSE
total_squared_error = 0
for i in range(0, len(validation_vector)):
    difference = validation_classes[i] - validation_vector[i]
    squared_error = math.pow(difference, 2)
    total_squared_error += squared_error
batch_se = total_squared_error / len(validation_vector)
rmse = math.sqrt(batch_se)
print("\nRMSE\n", rmse)

# SMAPE
total = 0
for i in range(0, len(validation_vector)):
    prediction = validation_vector[i]
    actual = validation_classes[i]
    difference = abs(actual - prediction)
    sample_smape = difference / (abs(actual) + abs(prediction))
    total += sample_smape
smape = (total / len(validation_vector))
print("\nSMAPE\n", smape)
print()
