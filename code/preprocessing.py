import numpy as np
import csv
import sys
# np.set_printoptions(threshold=sys.maxsize)

# Lookup Tables
# Index 11-LandSlope
LandSlope = {  
    "Gtl": 1,
    "Mod": 2,
    "Sev": 3
}
#index 27-ExterQual 28-ExterCond 30-BsmtQual 31-BsmtCond 40-HeatingQC 
# 53-KitchenQual 57-FireplaceQu 63-GarageQual 64-GarageCond 72-PoolQC
Quality = { 
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
    "NA": 0
}
# 32-BsmtExposure
BsmtExposure = {
    "Gd": 3,
    "Av": 2,
    "Mn": 1,
    "No": 0,
    "NA": 0
}


def getContinousData(zscore=True):
    data = []
    with open('train.csv') as realtordata:

        # read in data
        real_read = csv.reader(realtordata)

        # remove first row and column from data
        temp_data = []
        for row in real_read:
            temp_data.append(row)
        temp_data = np.array(temp_data)
        temp_data = temp_data[1:, 1:]

        # determine the total number of unique possible values for each feature
        # also store the possible values that those features can take on
        num_of_unique_values = np.zeros(np.size(temp_data, axis=1))
        possible_values = []
        temp_data_transpose = np.transpose(temp_data)
        for index in range(0, (np.size(temp_data_transpose, axis=0))):
            junk, counts = np.unique(
                temp_data_transpose[index], return_counts=True)
            num_of_unique_values[index] = len(counts)
            possible_values.append(junk)

        # loop through each row of data
        for row in temp_data:

            # use temp array to store the update feature values for this row
            temp = []

            # look through each value in the row
            for index in range(0, len(row)):

                value = None

                # skip the features that contain both integers and strings
                if index in [2,25,58]:  # LotFrontage,# MasVnrArea # GarageYrBlt
                    continue
                
                # if it can be converted to an integer, append it to the temp array
                try:
                    value = int(row[index])
                    temp.append(value)
                # if its a string, one hot encode it and then append the 1xD row to the temp array, value by value
                except:
                    value = row[index]
                    if index ==10 :
                        temp.append(LandSlope[value])
                    elif index in [26,27,29,30,39,52,56,62,63,71]:
                        temp.append(Quality[value])
                    elif index == 31:
                        temp.append(BsmtExposure[value])
                    else:
                        different_values = int(num_of_unique_values[index])
                        array = np.zeros(different_values)
                        for j in range(0, len(possible_values[index])):
                            if value == possible_values[index][j]:
                                array[j] = 1
                        for x in array:
                            temp.append(x)

            # append the new array to the data array
            data.append(temp)

    # separate classes from data
    data = np.array(data, dtype=int)
    classes = data[:, -1:]
    data = data[:, :-1]

    # z-scoring training data
    if zscore == True:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=1)
        data = (data - mean) / std

    # append classes back onto data
    data = np.append(data, classes, axis=1)

    return data


def getCategoricalData():
    data = []
    with open('train.csv') as realtordata:

        # read in data
        real_read = csv.reader(realtordata)

        # remove first row and column from data
        temp_data = []
        for row in real_read:
            temp_data.append(row)
        temp_data = np.array(temp_data)
        temp_data = temp_data[1:, 1:]

        # index of each continous feature within the data
        continuous_indexes = [3, 33, 35, 36, 37, 42,43, 44, 45, 61, 65, 66, 67, 68, 69, 70, 74]

        # get the mean for each continuous index
        means = []
        # for each continous feature index
        for cont_index in continuous_indexes:
            # gather all feature values from each sample
            values = []
            for sample in temp_data:
                value = int(sample[cont_index])
                values.append(value)
            # calculate the mean
            mean = np.sum(values) / len(values)
            # add the mean to the means array
            means.append(mean)

        # loop through each row of data
        for row in temp_data:

            # use temp array to store the updated feature values for this row
            temp = []

            # look through each value in the row
            for index in range(0, len(row)):

                # skip the features that contain both integers and strings
                if index in [2,4,5,25,58,74]:  # LotFrontage  25:# MasVnrArea 58:# GarageYrBlt 74:MiscVal
                    continue
                
                # get the actual value of the feature
                value = row[index]

                # check if the current feature index is equivalent to a continous feature index
                for cont_index in continuous_indexes:
                    if index == cont_index:
                        # if it is, convert to binary feature
                        value = int(value)
                        # get the mean for the continous feature
                        mean = means[continuous_indexes.index(index)]
                        if value < mean:
                            value = '0'
                        else:
                            value = '1'

                # append this to temporary row/sample array
                temp.append(value)

            # append the new row/sample array to the main data array
            data.append(temp)

    # separate classes from data
    data = np.array(data)
    classes = data[:, -1:]
    classes = np.array(classes, dtype=int)
    bin_classes = classify_in_bins(classes)
    
    assert classes.shape == bin_classes.shape
    
    unique_values = []
    data = np.array(data)
    for i in range(data.shape[1]):
        unique_values.append(np.unique(data[:,i]))
    return unique_values, bin_classes, data

def classify_in_bins(classes, binSize = 5000):
    new_classes = []
    for i in classes:
        try:
            value = int(i)
            nc = (round(value/binSize) + 1 )*binSize
            new_classes.append(nc)
        except:
            new_classes.append(0)
    return np.atleast_2d(new_classes).T

getCategoricalData()