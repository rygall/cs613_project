import numpy as np
import csv
import sys
np.set_printoptions(threshold=sys.maxsize)


def getBinWidth(data, feature_index, num_of_bins):

    return getBinWidth


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
            junk, counts = np.unique(temp_data_transpose[index], return_counts=True)
            num_of_unique_values[index] = len(counts)
            possible_values.append(junk)

        # loop through each row of data
        for row in temp_data:
            
            # use temp array to store the update feature values for this row
            temp = []

            # look through each value in the row
            for index in range(0, len(row)):

                value = None

                '''
                THIS COULD POTENITALLY BE AN EASIER SOLUTION BUT REMOVES WHOLE SAMPLES RATHER THAN FEATURES
                if value == 'NA':
                    break
                '''
                if index == 2: #LotFrontage
                    continue
                if index == 25: # MasVnrArea
                    continue
                if index == 58: # GarageYrBlt
                    continue

                # if it can be converted to an integer, append it to the temp array
                try:
                    value = int(row[index])
                    temp.append(value)
                # if its a string, one hot encode it and then append the 1xD row to the temp array, value by value
                except:
                    value = row[index]
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

    # get the means of the 
    continuous_data = getContinousData()
    means = np.mean(continuous_data, axis=0)
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

        continuous_indexes = [2, 3, 25, 33, 35, 36, 37, 42, 43, 44, 45, 61, 65, 66, 67, 68, 69, 70, 74]

        # loop through each row of data
        for row in temp_data:
            
            # use temp array to store the update feature values for this row
            temp = []

            # look through each value in the row
            for index in range(0, len(row)):
                
                # skip the features that contain both integers and strings
                if index == 2: #LotFrontage
                    continue
                if index == 25: # MasVnrArea
                    continue
                if index == 58: # GarageYrBlt
                    continue

                value = row[index]

                # check if the current feature index is equivalent to a continous feature index
                for cont_index in continuous_indexes:
                    if index == cont_index:
                        value = int(value)
                        if value < means[index]:
                            value = '0'
                        else:
                            value = '1'

                temp.append(value)


            # append the new array to the data array
            data.append(temp)

    # separate classes from data
    data = np.array(data)
    classes = data[:, -1:]
    data = data[:, :-1]

    # append classes back onto data
    data = np.append(data, classes, axis=1)

    print(data)
    return data