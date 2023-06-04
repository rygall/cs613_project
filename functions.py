import numpy as np
from collections import deque
import sys
import math


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
                
                    
# decision tree function
def myDT(samples, classes, num_classes=0, features=[], parent_classes=None):

    if num_classes == 0:
        values = np.unique(classes)
        num_classes = len(values)
    
    # this just creates an array of non-zero values to track what features have been used
    if len(features) == 0:
        features = np.zeros(np.size(samples, axis=1), dtype=int)

    # if all samples is empty, return most common class
    if len(samples) == 0:
        values, counts = np.unique(parent_classes, return_counts=True)
        index = np.argmax(counts)
        return values[index]
            
    # if all samples have the same class, return that class
    elif all(y == classes[0] for y in classes):
        return classes[0]

    # if all features have been used return the most frequent class
    elif all(x == 1 for x in features):
        values, counts = np.unique(classes, return_counts=True)
        index = np.argmax(counts)
        return values[index]

    # get average weighted entropy for each feature
    weighted_average_entropies = theirEntropies(samples, classes, num_classes)

    # finding minimum entropy feature
    min_entropy_feature_index = None
    min_entropy = 1
    for i in range(0, np.size(weighted_average_entropies, axis=0)):
        if (weighted_average_entropies[i] <= min_entropy):
            if (features[i] != 1):
                min_entropy = weighted_average_entropies[i]
                min_entropy_feature_index = i  
    
    # make the entropy for that index 1, so it is now not considered as a feature
    features[min_entropy_feature_index] = 1

    # create a node for the minimum feature index
    treeRoot = decisionTreeNode(min_entropy_feature_index)
    tree = decisionTree(treeRoot)

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
        subtree = myDT(value_samples, value_classes, num_classes, features, classes)
        tree.attach(value, subtree)

    return tree


# returns weighted average entropy of all features
def theirEntropies(samples, classes, num_classes):
    
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
                entropy = -( (value_node[3]/value_node[2]) * math.log((value_node[3]/value_node[2]), num_classes) )
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
                

# k-nearest neighbors function
def myKNN(training_data, training_classes, validation_data, k):

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

        # determine most frequent class
        values = np.unique(training_classes)
        counts = np.zeros(len(values))
        for node in linked_list:
            for i in range(len(values)):
                if node[1] == values[i]:
                    counts[i] += 1
        
        # determine highest count class
        max_class_index = np.argmax(counts)
        max_class = values[max_class_index]
        validation_vector.append(max_class)
    
    return validation_vector

        
