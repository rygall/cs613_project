
import numpy as np
import math
from preprocessing import Data
import random
import sys

class decisionTreeNode:
    def __init__(self, index):
        self.index = index

class decisionTree:
    def __init__(self, root):
        # Root Contains the Attribute Index of the attribute on which tree is built
        self.root = root
        # Branches contain the Brach Label (Attribute Values for attribute in Root) and Subtree or Leaf node
        self.branches = []

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

    
class RandomForst():
    def __init__(self,preProcessor: Data,num_trees=10,bin_classification=False,epochs=100000,eta=0.01) -> None:
        self.num_trees = num_trees
        self.preProcessor = preProcessor
        #Control Bin Classification Training
        self.bin_classification = False
        self.epochs = epochs
        self.eta = eta

    def process(self):
        # opening and reading data from data file
        feature_unique_values,bin_classes,data = self.preProcessor.getCategoricalData()
        data = np.array(data)

        # random shuffle data
        np.random.seed(0)
        np.random.shuffle(data)

        # splitting training and validation data
        split_index = int((2/3) * np.size(data, axis=0))
        self.training_data = data[:split_index, :-1]
        self.training_prices = data[:split_index, -1]
        self.training_classes = bin_classes[:split_index, -1]
        assert self.training_data.shape[0] == self.training_classes.shape[0]
        self.validation_data = data[split_index:, :-1]
        self.validation_prices = np.array(data[split_index:, -1],dtype=int)
        self.validation_classes = bin_classes[split_index:, -1]
        assert self.validation_data.shape[0] == self.validation_classes.shape[0]
        
        M = int(len(self.training_data)/2)
        print("Generating Random Forest...")
        # get random forest
        random_forest = self.myRF(M,feature_unique_values)
        self.validation(self.training_data,self.training_classes,self.training_prices,random_forest)        
        if self.bin_classification:
            training_tree_predictions = np.array(self.tree_predictions)*self.preProcessor.binSize
        else:
            training_tree_predictions = np.array(self.tree_predictions)
        self.validation(self.validation_data,self.validation_classes,self.validation_prices,random_forest)
        if self.bin_classification:
            validation_tree_predictions = np.array(self.tree_predictions)*self.preProcessor.binSize
        else:
            validation_tree_predictions = np.array(self.tree_predictions)
        print("Validation done using Random Forest, running linear regression on reults...")
        VYHat = self.linear_regresssion(training_tree_predictions,self.training_prices,validation_tree_predictions)

        # RMSE
        total_squared_error = 0
        for i in range(0, len(VYHat)):
            difference = self.validation_prices[i] - VYHat[i]
            squared_error = math.pow(difference, 2)
            total_squared_error += squared_error
        batch_se = total_squared_error / len(VYHat)
        rmse = math.sqrt(batch_se)
       

        # SMAPE
        total = 0
        for i in range(0, len(VYHat)):
            prediction = VYHat[i]
            actual = self.validation_prices[i]
            difference = abs(actual - prediction)
            sample_smape = difference / (abs(actual) + abs(prediction))
            total += sample_smape
        smape = (total / len(VYHat))
        print(f"Final RMSE={rmse} SMAPE={smape}")
        
    def linear_regresssion(self, X,Y,VX):
            max_epochs = self.epochs
            learning_rate = self.eta 
            # insert bias feature            
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0, ddof=1)
            X = (X - mean) / std
            X = np.insert(X, 0, 1, axis=1)
            Y = np.array(Y,dtype=int)
            VX = np.array(VX,dtype=int)
            VX = (VX - mean) / std
            VX = np.insert(VX, 0, 1, axis=1)
            # generate random weights
            np.random.seed(0)
            w = np.random.uniform(low=-0.0001, high=0.0001, size=np.size(X, axis=1))

            for epoch in range(0, max_epochs):

                    # determine all y_hats based on training data
                    y_hat = np.matmul(X, w)

                    # compute log loss for training data and add to tracking array
                    
                    # calculate the gradient
                    r2 = np.transpose(X)
                    r3 = np.subtract(y_hat, Y)
                    r4 = np.matmul(r2, r3)
                    n = np.size(y_hat, axis=0)
                    gradient = (2 / n) * r4
                    
                    # update weights based on average gradient
                    w = np.subtract(w, (learning_rate*gradient))

            validation_vector = np.matmul(VX, w)
            return validation_vector            

    # returns weighted average entropy of all features
    def getEntropies(self,samples, classes,features):        
        # loop through each feature
        weighted_average_entropies = []
        for i in features:
            
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
            
        weighted_average_entropies = np.array(weighted_average_entropies)
        features = np.array(features)
        # Return Features sorted by Entropy
        return features[np.argsort(weighted_average_entropies)]
                

    def myDT(self,samples, classes, features, feature_values=[], parent_classes=None):
        
        # if all features are used, return most common class
        if len(features) == 0:
            int_values = classes.astype(int)
            return round(np.mean(int_values))

        # if all samples is empty, return most common class
        if len(samples) == 0:
            int_values = parent_classes.astype(int)
            return round(np.mean(int_values))

        # if all samples have the same class, return that class
        if all(y == classes[0] for y in classes):
            return classes[0]

        # get random features
        if len(features) < 3:
            num_random = 1
        else:
            num_random = int(len(features) / 3)
        random_features = []
        num_random = 2
        for i in range(0, num_random):
            # get random feature
            feat_index = random.randint(0, len(features)-1)
            # add the random feature index to the new feature index array
            random_features.append(features[feat_index])
        
        features = list(features)
        # get average weighted entropy for each feature
        entropy_sorted_features = self.getEntropies(samples, classes,random_features)

        # finding minimum entropy feature
        min_entropy_feature_index = entropy_sorted_features[0]
        

        # create a node for the minimum entropy feature index
        treeRoot = decisionTreeNode(min_entropy_feature_index)
        tree = decisionTree(treeRoot)

        # delete feature index from availabele features index array
        del_index = features.index(min_entropy_feature_index)
        features = np.delete(features, del_index)

        # for each value of best feature
        for value in feature_values[min_entropy_feature_index]:
            # gather all samples that have the best feature as the current value
            value_samples = []
            value_classes = []
            for k in range(0, np.size(samples, axis=0)):
                if samples[k][min_entropy_feature_index] == value:
                    value_samples.append(samples[k])
                    value_classes.append(classes[k])
            value_samples = np.array(value_samples)
            value_classes = np.array(value_classes)
            subtree = self.myDT(value_samples, value_classes, features, feature_values, classes)
            tree.attach(value, subtree)

        return tree

    def bagging(self,samples,classes,prices,M):
        #Return bag of samples with replacement
        index_list = np.random.choice(len(samples),M,replace=True)
        return samples[index_list],classes[index_list],prices[index_list]

    def myRF(self,M,feature_unique_values):
        
        samples = self.training_data
        if self.bin_classification:
            classes = self.training_classes
        else:
            classes = self.training_prices
        
        
        # array to store trees
        random_forest = []
        training_smapes = []
        explored_trees = self.num_trees*2
        # building feature tracking array
        features = []
        for i in range(0, np.size(samples, axis=1)):
            features.append(i)
        
        # creating N trees with samples/N samples
        for i in range(0, explored_trees):
            
            tree_samples, tree_classes,tree_prices = self.bagging(samples,classes,self.training_prices, M)
            tree = self.myDT(tree_samples, tree_classes, features,feature_unique_values)
            rmse,smape = self.validation(tree_samples,tree_classes,tree_prices,[tree])
            random_forest.append(tree)
            training_smapes.append(smape)
        training_smapes = np.asarray(training_smapes)
        sorted_indices = np.argsort(training_smapes)
        sorted_forest = [random_forest[i] for i in sorted_indices]
        return random_forest[:self.num_trees]

    def validation(self,validation_data,validation_classes,validation_prices,random_forest):
        # determine predictions for each validation sample using the random forest
        validation_vector = []
        validation_classes = np.array(validation_classes, dtype=float)
        validation_prices = np.array(validation_prices, dtype=float)
        tree_predictions = []
        for sample in validation_data:
            # get prediction from each decision tree
            temp_predictions = []
            for tree in random_forest:
                classification = tree.classify(sample)
                if classification == None:
                    continue
                # determine actual price value
                if self.bin_classification:
                    # If Bin Classification take average of base + next group, so if something is between 0-25k, its 12.5K
                    price = (classification + self.preProcessor.binSize)/2
                else:
                    price = int(classification)

                # append the price prediction to the temp array
                temp_predictions.append(price)

            tree_predictions.append(temp_predictions)
            if len(temp_predictions) == 0:
                validation_vector.append(100000)
                continue  

            count = 0
            for pred in temp_predictions:
                count += pred
            mean_prediction = count / len(temp_predictions)

            # append the price prediction to the validation vector
            validation_vector.append(mean_prediction)
        self.tree_predictions = tree_predictions
        # RMSE
        total_squared_error = 0
        for i in range(0, len(validation_vector)):
            difference = validation_prices[i] - validation_vector[i]
            squared_error = math.pow(difference, 2)
            total_squared_error += squared_error
        batch_se = total_squared_error / len(validation_vector)
        rmse = math.sqrt(batch_se)
        

        # SMAPE
        total = 0
        for i in range(0, len(validation_vector)):
            prediction = validation_vector[i]
            actual = validation_prices[i]
            difference = abs(actual - prediction)
            sample_smape = difference / (abs(actual) + abs(prediction))
            total += sample_smape
        smape = (total / len(validation_vector))
        return rmse,smape

def main():
    num_trees = 10
    binSize = 25000
    bin_classification = False
    epochs = 200000
    eta = 0.001
    if(len(sys.argv)>1):
        num_trees = int(sys.argv[1])
    if(len(sys.argv)>2):
        binSize = int(sys.argv[2])
        bin_classification = True
    if(len(sys.argv)>3):
        epochs = int(sys.argv[3])
    if(len(sys.argv)>3):
        eta = float(sys.argv[4])
        
    print(f"Running with Parameters:num_trees:{num_trees} binSize:{binSize} bin_classification:{bin_classification} epochs:{epochs} eta:{eta}")
    d = Data(binSize=binSize)
    rf = RandomForst(d,num_trees,bin_classification,epochs,eta)
    rf.process()
    

if __name__ == "__main__":
    main()