import numpy as np
import csv
import sys
#np.set_printoptions(threshold=sys.maxsize)


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

                # skip the features that contain both integers and strings
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
        continuous_indexes = [3, 33, 35, 36, 37, 42, 43, 44, 45, 61, 65, 66, 67, 68, 69, 70, 74]

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
                if index == 2: #LotFrontage
                    continue
                if index == 4: 
                    continue
                if index == 5:
                    continue
                if index == 25: # MasVnrArea
                    continue
                if index == 58: # GarageYrBlt
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

    '''
    # separate classes from data
    data = np.array(data)
    classes = data[:, -1:]
    data = data[:, :-1]
    classes = np.array(classes, dtype=int)

    ''''''
    # !! WE SET THIS VALUE !!
    number_of_bins = 100000

    # put prices into bins
    temp_classes = classes.flatten()
    temp_classes = np.array(temp_classes, dtype=int)
    price_range = np.ptp(temp_classes)
    bin_range = price_range / number_of_bins
    price_min = np.min(temp_classes)
    bins = []
    for i in range(0, number_of_bins):
        # bin = min value, max value
        bin = [i, 0, 0]
        bin[1] = price_min + (i * bin_range)
        bin[2] = price_min + ((i+1) * bin_range)
        bins.append(bin)

    # placing samples into bins
    new_classes = []
    for i in range(0, len(classes)):
        sample_price = int(classes[i])
        bin_assignment = 0
        for bin in bins:
            if bin[1] < sample_price <= bin[2]:
                bin_assignment = bin[0]
        new_classes.append(bin_assignment)
    

    # append classes back onto data
    new_classes = np.atleast_2d(new_classes).T
    data = np.append(data, new_classes, axis=1)
    '''

    return data

getCategoricalData()