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

data = np.array(data, dtype=float)

# END OF DATA PREPROCESSING


# BEGINNING OF S FOLDS LOOP

s_folds = [3, 223, 1338]
for s in s_folds:

    # 20x loop
    rsme_values = []
    for i in range(0, 20):

        # shuffling the rows of data
        iter_data = np.copy(data)
        np.random.seed(i)
        np.random.shuffle(iter_data)
    
        # breaking the data into folds
        folds = []
        folds_charges = []
        fold_size = int(np.size(iter_data, axis=0)/s)
        for j in range(0, s):
            fold = []
            fold_charges = []
            for k in range(0, fold_size):
                charge = []
                charge.append(iter_data[0][10])
                charge = np.array(charge)
                fold_charges.append(charge)
                row = iter_data[0]
                row = np.delete(row, 10, axis=0)
                fold.append(row)
                iter_data = np.delete(iter_data, 0, axis=0)
            fold = np.array(fold)
            folds.append(fold)
            fold_charges = np.array(fold_charges)
            folds_charges.append(fold_charges)
        folds = np.array(folds)
        folds_charges = np.array(folds_charges)

        # inner i=1 to S for loop
        sqaured_errors = []
        for z in range(0, s):
        
            # taking out i as the validation data
            validation_data = folds[z]
            validation_charges = folds_charges[z]
            training_data_folds = np.delete(folds, z, axis=0)
            training_charges_folds = np.delete(folds_charges, z, axis=0)
            
            # combine training folds into one 2D array for data and charges
            training_data = []
            for row in training_data_folds:
                for obs in row:
                    training_data.append(obs)
            training_data = np.array(training_data)
            training_charges = []
            for row in training_charges_folds:
                for obs in row:
                    training_charges.append(obs)
            training_charges = np.array(training_charges)
            
            # training model using direct solution
            x = training_data
            y = training_charges
            x_transpose = np.transpose(x)
            result_1 = np.matmul(x_transpose, x)
            inv_result_1 = np.linalg.pinv(result_1)
            result_2 = np.matmul(x_transpose, y)
            w = np.matmul(inv_result_1, result_2)

            # computing sqaured error for each sample
            # calculate yhats
            y_hats = []
            for obs in validation_data:
                y_hat = np.matmul(obs, w)
                y_hats.append(y_hat)
            y_hats = np.array(y_hats)
            # calculating y - yhat
            diff = np.subtract(validation_charges, y_hats)
            # calculating sqaured error
            se = np.power(diff, 2)
            # adding squared error to se array
            for e in se:
                sqaured_errors.append(e)
        sqaured_errors = np.array(sqaured_errors)
        # compute rmse for the squared errors
        sum = np.sum(sqaured_errors, axis=0)
        sum_divided_by_n = sum / ((np.size(sqaured_errors, axis=0)))
        rmse = np.sqrt(sum_divided_by_n)
        rsme_values.append(rmse)

    # compute mean and std deviation of the numbers
    mean = np.mean(rsme_values)
    std = np.std(rsme_values, ddof=1)

    print("=========================================")
    print("Folds =", s)
    print("RSME Mean =", mean)
    print("RSME Standard Deviation =", std)
    print("=========================================")


# END OF S FOLDS LOOP