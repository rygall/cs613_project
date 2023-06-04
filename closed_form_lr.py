import numpy as np
import csv
import sys
np.set_printoptions(threshold=sys.maxsize)


# BEGINNING OF DATA PREPROCESSING

# extracting data and pre processing it for ease of use in the algorithm
data = []
with open('insurance.csv') as insurance:
    ins_read = csv.reader(insurance)
    count = 0
    for row in ins_read:
        # skip column labels
        if count == 0:
            count = count + 1
            continue
        
        # we use the temp array to pre process the data for the algorithm
        temp = np.array(row[0:7])

        # convert gender into binary feature
        if temp[1] == 'male':
            temp[1] = 1
        else:
            temp[1] = 0

        # make smoker a binary feature
        if temp[4] == 'yes':
            temp[4] = 1
        else:
            temp[4] = 0

        # make region into 4 binary features
        # one-hot encoding
        region = temp[5]
        temp = np.delete(temp, 5)
        temp = np.append(temp, 0)
        temp = np.append(temp, 0)
        temp = np.append(temp, 0)
        temp = np.append(temp, 0)
        if region == 'northeast':
            temp[6] = 1
        if region == 'northwest':
            temp[7] = 1
        if region == 'southeast':
            temp[8] = 1
        if region == 'southwest':
            temp[9] = 1
    
        # moving charges to the end of the array
        temp = np.append(temp, temp[5])
        temp = np.delete(temp, 5)

        # inserting a bias feature into the front of the array
        temp = np.insert(temp, 0, 1)

        # append the new array to the data array
        data.append(temp)
        count = count + 1

# shuffling the rows of data
data = np.array(data, dtype=float)
np.random.seed(0)
np.random.shuffle(data)

# breaking the data out into training data and validation data
length = np.size(data, axis=0)
length = int((2/3)*length)
validation_data = data
training_data = []
for i in range(0, length):
    training_data.append(validation_data[0])
    validation_data = np.delete(validation_data, 0, axis=0)
training_data = np.array(training_data)

# removing the charges from training data
training_charges = []
for row in training_data:
    temp = []
    temp.append(row[10])
    training_charges.append(temp)
training_data = np.delete(training_data, 10, axis=1)
training_charges = np.array(training_charges)

# removing the charges from validation data
validation_charges = []
for row in validation_data:
    temp = []
    temp.append(row[10])
    validation_charges.append(temp)
validation_data = np.delete(validation_data, 10, axis=1)
validation_charges = np.array(validation_charges)

# END OF DATA PREPROCESSING


# BEGINNING OF MODEL TRAINING

x1 = training_data
y1 = training_charges

x_transpose_1 = np.transpose(x1)
result_1 = np.matmul(x_transpose_1, x1)
inv_result_1 = np.linalg.pinv(result_1)
result_2 = np.matmul(x_transpose_1, y1)
w = np.matmul(inv_result_1, result_2)

# END OF MODEL TRAINING


# BEGINNING CALCULATION OF RMSE and SMAPE

# calculate training yhats
y_hat_w1 = []
for obs in training_data:
    y_hat_1 = np.matmul(obs, w)
    y_hat_w1.append(y_hat_1)
y_hat_w1 = np.array(y_hat_w1)
# calculating y - yhat
training_diff = np.subtract(training_charges, y_hat_w1)
# calculating sqaured error
training_se = np.power(training_diff, 2)
# calculating sum of squared errors
training_sum = np.sum(training_se, axis=0)
# calculation sqaure root of the sum of sqaured errors
training_rmse = (np.sqrt(training_sum/(np.size(training_data, axis=0))))

# calculating validation yhats
y_hat_w2 = []
for obs in validation_data:
    y_hat_2 = np.matmul(obs, w)
    y_hat_w2.append(y_hat_2)
y_hat_w2 = np.array(y_hat_w2)
# calculating y - yhat
validation_diff = np.subtract(validation_charges, y_hat_w2)
# calculating sqaured error
validation_se = np.power(validation_diff, 2)
# calculating sum of squared errors
validation_sum = np.sum(validation_se, axis=0)
# calculation sqaure root of the sum of sqaured errors
validation_rmse = (np.sqrt(validation_sum/(np.size(validation_data, axis=0))))

print('Training RSME:\n', training_rmse[0])
print('Validation RSME:\n', validation_rmse[0])

# calculating training smape
training_smape = np.divide((np.abs(training_diff)), (np.abs(training_charges)+np.abs(y_hat_w1)))
training_smape = np.sum(training_smape, axis=0)
training_smape = np.divide(training_smape, np.size(training_data, axis=0))

# calculating validation smape
validation_smape = np.divide((np.abs(validation_diff)), (np.abs(validation_charges)+np.abs(y_hat_w2)))
validation_smape = np.sum(validation_smape, axis=0)
validation_smape = np.divide(validation_smape, np.size(validation_data, axis=0))

print('Training SMAPE:\n', training_smape[0])
print('Validation SMAPE:\n', validation_smape[0])

# END CALCULATION OF RMSE and SMAPE








