---
title: "Movement Classification"
author: "Niyazi Ülke"
date: "06/10/2020"
output: 
  html_document:
    keep_md: true
---


## Overview

This is a course project from Coursera Reproducible Research MOOC by JHU. In this project, data from accelometers on the belt, forearm, arm and dumbell of 6 participants will be used. Source of data is :
&nbsp;
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har 

and detailed information can be found at the link. The aim is to predict how well the participants do some physical activites. __"classe"__ variable stores this information. This is a classification problem.

## Data Exploration 

Import caret library and download the files if it is needed. Read csv files.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
if(!file.exists("pml-training.csv")){
  download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="pml-training.csv")
}

if(!file.exists("pml-testing.csv")){
  download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="pml-testing.csv")
}  

unPartitionedSet = read.csv(file = "pml-training.csv" ,stringsAsFactors =T,header = T)
testingSet = read.csv(file = "pml-testing.csv", stringsAsFactors =T, header = T)
```

Print sizes of the data sets.

```r
dim(unPartitionedSet)
```

```
## [1] 19622   160
```

```r
dim(testingSet)
```

```
## [1]  20 160
```

Investigate NA values.


```r
naControl = colSums(is.na(unPartitionedSet))
unique(naControl)
```

```
## [1]     0 19216
```

## Data Preprocessing

Columns contain either 0 or 19216 NA values. Columns which contain NA's must be excluded from the data.
Also the first 7 variables are not related to our model. They contain generic knowledge.


```r
unPartitionedSet = unPartitionedSet[, -c(1:7)]
unPartitionedSet = unPartitionedSet[, colSums(is.na(unPartitionedSet))==0]
```

Create data partition with 70%.

```r
set.seed(111)
trainIndices = createDataPartition(unPartitionedSet$classe,p=0.7)[[1]]
trainingSet = unPartitionedSet[trainIndices,]
validationSet = unPartitionedSet[-trainIndices,]
```

Variables with little variance do not have significant impact.
They should be excluded from training set.


```r
lowVar = nearZeroVar(trainingSet)
trainingSet = trainingSet[,-lowVar]
```

Check new dimensions.


```r
dim(trainingSet)
```

```
## [1] 13737    53
```
With data preparation, number of variables in training set is reduced from 160 to 53

&nbsp;

Examine the classe variable. (The project aims to predict it).

```r
str(trainingSet$classe)
```

```
##  Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```


```r
table(trainingSet$classe)
```

```
## 
##    A    B    C    D    E 
## 3906 2658 2396 2252 2525
```
## Applying Machine Learning Algorithms

Threefold cross validation will be applied.

```r
threefold = trainControl(method="cv", number=3, verboseIter=FALSE)
```

### KNN

```r
set.seed(333)
knnModel = train(classe ~ ., data = trainingSet, method = "knn", trControl=threefold)
predictKnn = predict(knnModel, newdata=validationSet)
confusionMatrix(predictKnn, as.factor(validationSet$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1596   54   14   22   11
##          B   27  962   35    6   37
##          C   15   56  944   52   30
##          D   29   36   17  867   28
##          E    7   31   16   17  976
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9082          
##                  95% CI : (0.9006, 0.9155)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8839          
##                                           
##  Mcnemar's Test P-Value : 1.275e-09       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9534   0.8446   0.9201   0.8994   0.9020
## Specificity            0.9760   0.9779   0.9685   0.9776   0.9852
## Pos Pred Value         0.9405   0.9016   0.8605   0.8874   0.9322
## Neg Pred Value         0.9814   0.9633   0.9829   0.9802   0.9781
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2712   0.1635   0.1604   0.1473   0.1658
## Detection Prevalence   0.2884   0.1813   0.1864   0.1660   0.1779
## Balanced Accuracy      0.9647   0.9112   0.9443   0.9385   0.9436
```
This model has 91% accuracy.

### Decision Tree

```r
set.seed(333)
decisionTreeModel = train(classe ~ ., data=trainingSet, method="ctree", trControl=threefold)
predictDecisionTree = predict(decisionTreeModel, newdata=validationSet)
confusionMatrix(predictDecisionTree, as.factor(validationSet$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1562   52    8   34    8
##          B   51 1015   62   35   29
##          C   19   36  917   37   23
##          D   28   23   23  849   24
##          E   14   13   16    9  998
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9076          
##                  95% CI : (0.8999, 0.9148)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8831          
##                                           
##  Mcnemar's Test P-Value : 0.0002222       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9331   0.8911   0.8938   0.8807   0.9224
## Specificity            0.9758   0.9627   0.9763   0.9801   0.9892
## Pos Pred Value         0.9387   0.8515   0.8886   0.8965   0.9505
## Neg Pred Value         0.9735   0.9736   0.9775   0.9767   0.9826
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2654   0.1725   0.1558   0.1443   0.1696
## Detection Prevalence   0.2828   0.2025   0.1754   0.1609   0.1784
## Balanced Accuracy      0.9544   0.9269   0.9350   0.9304   0.9558
```
This model has 91% accuracy too.

### Random Forest

Train random forest with 100 trees.

```r
set.seed(333)
randomForestModel = train(classe ~ ., data=trainingSet, method="rf", ntree=100 ,trControl=threefold)
predictRandomForest = predict(randomForestModel$finalModel, newdata=validationSet)
confusionMatrix(predictRandomForest, as.factor(validationSet$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674   10    0    0    0
##          B    0 1126    4    0    2
##          C    0    3 1020   12    3
##          D    0    0    2  951    7
##          E    0    0    0    1 1070
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9925        
##                  95% CI : (0.99, 0.9946)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9905        
##                                         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9886   0.9942   0.9865   0.9889
## Specificity            0.9976   0.9987   0.9963   0.9982   0.9998
## Pos Pred Value         0.9941   0.9947   0.9827   0.9906   0.9991
## Neg Pred Value         1.0000   0.9973   0.9988   0.9974   0.9975
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1913   0.1733   0.1616   0.1818
## Detection Prevalence   0.2862   0.1924   0.1764   0.1631   0.1820
## Balanced Accuracy      0.9988   0.9937   0.9952   0.9923   0.9944
```

Random forest has the highest accuracy with 99% . This model will be used for predicting the test set.

## Predicting

```r
predict(randomForestModel, newdata = testingSet)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The results are predictions on test set. This model is 100% successful at predicting the test set. Note: The test set does not contain __'classe'__  variable to compare, there is a quiz in the MOOC to examine actual values.
