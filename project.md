# Coursera: Practical Machine Learning Prediction Assignment
Nawal Sarda [GitHub](https://github.com/nbsarda)  


```
## Run time: 2016-03-10 10:37:06
## R version: R version 3.2.3 (2015-12-10)
```

Assignment:
The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Data Preprocessing
-------------------------

```r
# load libraries
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
# load the training set
pml.training <- read.csv("pml-training.csv", na.strings=c("NA",""))

# load the testing set
# Note: the testing set is not used in this analysis
# the set is only used for the second part of the assignment
# when the model is used to predict the classes
pml.testing <- read.csv("pml-testing.csv", na.strings=c("NA",""))

# summary(pml.training)
```
We are interested in variables that predict the movement
The set contains a number of variables that can be removed: 
* X (= row number)
* user_name (= the name of the subject)

cvtd_timestamp is removed because it is a factor instead of a numeric value
and the raw_timestamp_part_1 + raw_timestamp_part_2 contain the same info
in numeric format.

```r
rIndex <- grep("X|user_name|cvtd_timestamp",names(pml.training))
pml.training <- pml.training[, -rIndex]
```
Some variable have near Zero variance which indicates that 
they do not contribute (enough) to the model.
They are removed from the set.

```r
nzv <- nearZeroVar(pml.training)
pml.training <- pml.training[, -nzv]
```
A number of variable contain (a lot of) NA's.
Leaving them in the set not only makes the model creation slower, but also results in lower accuracy in the model.
These variables will be removed from the set:

```r
NAs <- apply(pml.training,2,function(x) {sum(is.na(x))}) 
pml.training <- pml.training[,which(NAs == 0)]
```

The original set is rather large (19622 obs. of 56 variables). 
We create a smaller training set of 80% of the original set

```r
tIndex <- createDataPartition(y = pml.training$classe, p=0.2,list=FALSE) 
pml.sub.training <- pml.training[tIndex,] # 3927 obs. of 56 variables
pml.test.training <- pml.training[-tIndex,] # test set for cross validation
```


Model creation
-------------------------
We can now create a model based on the pre-processed data set. 
Note that at this point, we are still working with a large set of variables.
We do have however a reduced number of rows.

A first attempt to create a model is done by fitting a single tree:

```r
modFit <- train(pml.sub.training$classe ~.,data=pml.sub.training,method="rpart")
```

```
## Loading required package: rpart
```

```r
modFit
```

```
## CART 
## 
## 3927 samples
##   55 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 3927, 3927, 3927, 3927, 3927, 3927, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03948773  0.5720107  0.45190429  0.03925126   0.05459532
##   0.04423100  0.5620400  0.43751003  0.03919020   0.05496794
##   0.10921380  0.3240386  0.05922407  0.04008987   0.05829062
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03948773.
```

```r
results <- modFit$results
round(max(results$Accuracy),4)*100
```

```
## [1] 57.2
```
Note that running the train() function can take some time!
The accuracy of the model is low: 57.2 %

A second attempt to create a model is done by using Random forests:

```r
ctrl<- trainControl(method = "cv", number =4, allowParallel = TRUE)
modFit <- train(pml.sub.training$classe ~.,data = pml.sub.training,method="rf",prof=TRUE, trControl = ctrl)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
modFit
```

```
## Random Forest 
## 
## 3927 samples
##   55 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 2945, 2945, 2945, 2946 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9714818  0.9639116  0.008753130  0.011081714
##   28    0.9875229  0.9842142  0.003657691  0.004627672
##   55    0.9849768  0.9809927  0.005080300  0.006427634
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```

```r
results <- modFit$results
round(max(results$Accuracy),4)*100
```

```
## [1] 98.75
```
This second attempt provides us with a model that has a much higher accuracy: : 98.75 %

Cross-validation
-------------------------

We now use the modFit to predict new values within the test set that we created for cross-validation:

```r
pred <- predict(modFit,pml.test.training)
pml.test.training$predRight <- pred==pml.test.training$classe
table(pred, pml.test.training$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 4463   12    0    0    0
##    B    1 3008   22    0    5
##    C    0   17 2706   18    0
##    D    0    0    9 2546   43
##    E    0    0    0    8 2837
```
As expected the predictions are not correct in all cases.
We can calculate the accuracy of the prediction:

```r
pRes <- postResample(pred, pml.test.training$classe)
pRes
```

```
##  Accuracy     Kappa 
## 0.9913985 0.9891200
```
The prediction fitted the test set even slightly better than the training set: 99.1399 %

### Expected out of sample error
We can calculate the expected out of sample error based on the test set that we created for cross-validation:

```r
cfM <- confusionMatrix(pred, pml.test.training$classe)
cfM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4463   12    0    0    0
##          B    1 3008   22    0    5
##          C    0   17 2706   18    0
##          D    0    0    9 2546   43
##          E    0    0    0    8 2837
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9914          
##                  95% CI : (0.9898, 0.9928)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9891          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9998   0.9905   0.9887   0.9899   0.9834
## Specificity            0.9989   0.9978   0.9973   0.9960   0.9994
## Pos Pred Value         0.9973   0.9908   0.9872   0.9800   0.9972
## Neg Pred Value         0.9999   0.9977   0.9976   0.9980   0.9963
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1917   0.1724   0.1622   0.1808
## Detection Prevalence   0.2851   0.1934   0.1746   0.1655   0.1813
## Balanced Accuracy      0.9994   0.9941   0.9930   0.9930   0.9914
```
The expected out of sample error is: 0.8601 % 

Note: The confusionMatrix function from the Caret package does provide all the information that we calculated 'by hand' in the first part of the Cross-validation. It shows that both methods provide the same answer.




