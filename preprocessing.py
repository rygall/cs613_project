import numpy as np
import csv
import sys
#np.set_printoptions(threshold=sys.maxsize)


# BEGINNING OF DATA PREPROCESSING

# extracting data and pre processing it for ease of use in the algorithm
data = []
with open('dataset/train.csv') as realtordata:
    real_read = csv.reader(realtordata)
    count = 0
    for row in real_read:
        print(row)
        # skip column labels row
        if count == 0:
            count = count + 1
            continue
        
        # we use the temp array to preprocess each row
        temp = np.array(row[0:10])
        print(temp)



        '''
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
        '''

        # inserting a bias feature into the front of the array
        temp = np.insert(temp, 0, 1)

        # append the new array to the data array
        data.append(temp)
        count = count + 1

data = np.array(data, dtype=float)

# END OF DATA PREPROCESSING