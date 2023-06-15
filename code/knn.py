import numpy as np
from collections import deque
import sys
import math
from preprocessing import Data

class knn():
    def __init__(self, k):
        self.k = k

    # k-nearest neighbors function
    def myKNN(self, training_data, training_classes, validation_data, k):

        # loop through each validation sample and determine nearest neighbor class
        validation_vector = []
        for validation_sample in validation_data:

            # linked list to store nearest neighbors
            linked_list = deque()
            for x in range(0, k):
                linked_list.append([sys.maxsize, 1])
            

            # get nearest neighbors
            for i in range(0, len(training_data)):
                squared_euclidian = np.subtract(validation_sample, training_data[i])
                squared_euclidian = np.power(squared_euclidian, 2)
                squared_euclidian = np.sum(squared_euclidian) 
                euclidian = np.sqrt(squared_euclidian)
                new_node = [euclidian, training_classes[i]]
                linked_list.append(new_node)
                max_node = new_node
                for node in linked_list:
                    if node[0] > max_node[0]:
                        max_node = node
                linked_list.remove(max_node)

            # calculate mean prediction
            total = 0
            for node in linked_list:
                total += node[1]
            prediction = total / len(linked_list)

            # determine highest count class
            validation_vector.append(prediction)
        
        return validation_vector

    def process(self):
        # opening and reading data from data file
        preprocesser = Data()
        data = preprocesser.getContinousData()

        # random shuffle data
        np.random.seed(0)
        np.random.shuffle(data)

        # splitting training and validation data
        split_index = int((2/3) * np.size(data, axis=0))
        training_data = data[:split_index, :-1]
        training_classes = data[:split_index, -1]
        validation_data = data[split_index:, :-1]
        validation_classes = data[split_index:, -1]

        # get validation vector
        validation_vector = self.myKNN(training_data, training_classes, validation_data, self.k)

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

def main():
    k=5
    if(len(sys.argv)>1):
        k = int(sys.argv[1])
    nn = knn(k)
    nn.process()

if __name__ == "__main__":
    main()

