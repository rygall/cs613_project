import numpy as np
import math
import preprocessing as data
import random
import sys


class decisionTreeNode:
    def __init__(self, index):
        self.index = index



class decisionTree:
    def __init__(self, root):
        self.root = root
        self.branches = []

    def root(self):
        return self.root
    
    def attach(self, value, dt):
        branch_node = [value, dt]
        self.branches.append(branch_node)

    def classify(self, sample):
        for branch in self.branches:
            if sample[self.root.index] == branch[0]:
                try:
                    return branch[1].classify(sample)
                except:
                    return branch[1]

    

# returns weighted average entropy of all features
def theirEntropies(samples, classes):
    
    # loop through each feature
    weighted_average_entropies = []
    for i in range(0, np.size(samples, axis=1)):
        
        # determine all possible values the feature can take on
        values_nodes = []
        for j in range(0, np.size(samples, axis=0)):
            sample_feature_value = samples[j][i]
            sample_class = classes[j]
            new_value = True
            new_class = True
            num_of_instances = 1
            # each node = [feature, class, number of total nodes with that value, number of total nodes with that value and class]
            for k in range(0, np.size(values_nodes, axis=0)):
                node = values_nodes[k]
                if sample_feature_value == node[0]:
                    new_value = False
                    node[2] = node[2] + 1
                    num_of_instances = node[2]
                    if sample_class == node[1]:
                        new_class = False
                        node[3] = node[3] + 1
            if (new_class == True):
                new_node = [sample_feature_value, sample_class, num_of_instances, 1]
                values_nodes.append(new_node)
            elif (new_value == True): 
                new_node = [sample_feature_value, sample_class, 1, 1]
                values_nodes.append(new_node)

        # determine entropy of each value/class pair
        entropy_nodes = []
        for value_node in values_nodes:
            # calculate the entropy
            value = value_node[0]
            entropy = 0
            if value_node[3] == value_node[2]:
                entropy = 0
            else:
                entropy = -( (value_node[3]/value_node[2]) * math.log((value_node[3]/value_node[2]), 2) )
            # add the entropy to its respective value entropy node
            new_value = True
            for k in range(0, len(entropy_nodes)):
                if entropy_nodes[k][0] == value:
                    entropy_nodes[k].append(entropy)
                    new_value = False
                    break
            if new_value == True:
                new_node = [value, value_node[2], entropy]
                entropy_nodes.append(new_node)

        # for each entropy multiply it by the ratio of the number of instances of the value over the total number of samples
        weighted_entropies = []
        for entropy_node in entropy_nodes:
            total_entropy = 0
            for x in range(2, len(entropy_node)):
                total_entropy += entropy_node[x]
            weighted_entropy = (entropy_node[1] / np.size(samples, axis=0)) * total_entropy
            weighted_entropies.append(weighted_entropy)
        weighted_average_entropy = np.sum(weighted_entropies)
        weighted_average_entropies.append(weighted_average_entropy)


    return weighted_average_entropies
                


def myDT(samples, classes, features, parent_classes=None, num_classes=0):
    
    # if all features are used, return most common class
    if len(features) == 0:
        values, counts = np.unique(parent_classes, return_counts=True)
        index = np.argmax(counts)
        return values[index]

    # if all samples is empty, return most common class
    if len(samples) == 0:
        values, counts = np.unique(parent_classes, return_counts=True)
        index = np.argmax(counts)
        return values[index]
            
    # if all samples have the same class, return that class
    elif all(y == classes[0] for y in classes):
        return classes[0]

    # get random features
    if len(features) <= 6:
        num_random = len(features)
    else:
        num_random = int(len(features) / 3)
    random_features = []
    for i in range(0, num_random):
        # get random feature
        feat_index = random.randint(0, len(features)-1)
        # add the random feature index to the new feature index array
        random_features.append(features[feat_index])
        # delete the select random feature from original feature index array
        features = np.delete(features, feat_index)
    features = random_features

    # get average weighted entropy for each feature
    weighted_average_entropies = theirEntropies(samples, classes)

    # finding minimum entropy feature
    min_entropy_feature_index = None
    min_entropy = sys.maxsize
    for i in range(0, np.size(weighted_average_entropies, axis=0)):
        if (weighted_average_entropies[i] <= min_entropy):
            if i in features:
                min_entropy = weighted_average_entropies[i]
                min_entropy_feature_index = i  

    # create a node for the minimum entropy feature index
    treeRoot = decisionTreeNode(min_entropy_feature_index)
    tree = decisionTree(treeRoot)

    # delete feature index from availabele features index array
    del_index = features.index(min_entropy_feature_index)
    features = np.delete(features, del_index)

    # determine all possible values the feature can take on
    values = []
    for j in range(0, np.size(samples, axis=0)):
        sample_feature_value = samples[j][min_entropy_feature_index]
        new_value = True
        for k in range(0, np.size(values, axis=0)):
            value = values[k]
            if sample_feature_value == value:
                new_value = False
        if (new_value == True): 
            values.append(sample_feature_value)

    # for each value of best feature
    for value in values:
        # gather all samples that have the best feature as the current value
        value_samples = []
        value_classes = []
        for k in range(0, np.size(samples, axis=0)):
            if samples[k][min_entropy_feature_index] == value:
                value_samples.append(samples[k])
                value_classes.append(classes[k])
        value_samples = np.array(value_samples)
        value_classes = np.array(value_classes)
        subtree = myDT(value_samples, value_classes, features, classes)
        tree.attach(value, subtree)

    return tree



def myRF(samples, classes, num_trees):
    
    # array to store trees
    random_forest = []

    # building feature tracking array
    features = []
    for i in range(0, np.size(samples, axis=1)):
        features.append(i)
    
    # creating N trees
    for i in range(0, num_trees):
        tree = myDT(samples, classes, features)
        random_forest.append(tree)

    return random_forest



# opening and reading data from data file
data, true_classes, bins = data.getCategoricalData()

# random shuffle data
np.random.seed(0)
np.random.shuffle(data)

# splitting training and validation data
split_index = int((2/3) * np.size(data, axis=0))
training_data = data[:split_index, :-1]
training_classes = data[:split_index, -1]
validation_data = data[split_index:, :-1]
validation_classes = data[split_index:, -1]

# !! SET NUMBER OF TREES HERE !!
# !! MUST BE ODD !!
num_trees = 1

# get random forest
random_forest = myRF(training_data, training_classes, num_trees)

# determine predictions for each validation sample using the random forest
validation_vector = []
for sample in validation_data:
    # get prediction from each decision tree
    temp_predictions = []
    for tree in random_forest:
        classification = tree.classify(sample)
        temp_predictions.append(classification)

    # determine actual price value
    prediction_index = np.argmax(temp_predictions)
    bin_prediction = None
    try:
        bin_prediction = int(temp_predictions[prediction_index])
    except:
        bin_prediction = 10
    price_prediction = None
    for x in bins:
        bin_value = int(x[0])
        if bin_prediction == bin_value:
            price_prediction = (x[2] + x[1]) / 2 
    # append the price prediction to the validation vector
    validation_vector.append(price_prediction)

# RMSE
total_squared_error = 0
for i in range(0, len(validation_vector)):
    difference = true_classes[i] - validation_vector[i]
    squared_error = math.pow(difference, 2)
    total_squared_error += squared_error
batch_se = total_squared_error / len(validation_vector)
rmse = math.sqrt(batch_se)
print("\nRMSE\n", rmse)

# SMAPE
total = 0
for i in range(0, len(validation_vector)):
    prediction = validation_vector[i]
    actual = true_classes[i]
    difference = abs(actual - prediction)
    sample_smape = difference / (abs(actual) + abs(prediction))
    total += sample_smape
smape = (total / len(validation_vector))
print("\nSMAPE\n", smape)
print()


