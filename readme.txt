 - CS613 Machine Learning
 - Submitted By Ambadas Kulkarni, Kenneth Kimble, Ryan Gallagher

 Submission Contents :
 ./code   - Contains Main Code 
    lr.py - Linear Regression
    knn.py - KNN 
    rf.py - Random Forest
    train.csv - Data file
./data - Data Folder - Only for reference
readme.txt - This readme

Executing the Algorithms: - Note that all code must be executed from "code" folder.
cd code 

1. Executing Linear Regression :

2. Executing KNN:
python3 knn.py {K}   
example : 
- K - KNN takes only one integer parameter K - not it must be a positive integer  Default Value K=5

3. Executing Random Forest:
python3 rf.py {num_trees} {binSize} {epochs} {eta}
num_trees : Number of trees to be used in forest. The Algorithm explores 2*num_trees, and selects the best num_trees
epochs : Number of Epochs to run on Linear Regression, which executed on RF results   Default = 200000
eta : Learning Rate for Linear Regression  Do not use a large number, since house prices are huge numbers, it might overflow. Default = 0.001
binSize : Bins here are price category. E.g. binSize = 25000, divides price into 0-25000,25000-50000 etc. Since prices are in the range of 100K, do not use a too small bin size 5K-50K should be sweet spot
          bin_classification:True - This is hidden parameter it's activated with binSize is overridden. False Otherwise

examples : With Bin - Classification.
1. python3 rf.py 30 200000 0.001 25000   # We found best results with this, might take a little time
2. python3 rf.py 15 200000 0.001 25000

Without Bin - Classification - In this case, The actual house price is considered the class
1. python3 rf.py  15 100000 0.0001    # Don't override binSize, that makes it consider House Price as class, you can still override num_trees,epochs,eta
2. python3 rf.py  10