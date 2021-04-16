---
title: "Prediction in Exercise Pattern behaviours"
author: "Wanda Ken - Date: 16/04/2021"
geometry: "left=.5cm,right=.5 cm,top=.6cm,bottom=1.1cm"
output:
  html_document: 
    keep_md: yes
  pdf_document: 
    fig_crop: no
    toc_depth: 1
fontsize: 11pt
---



## Overview
Data from accelerometers on the belt, forearm, arm and dumbell of 6 participants will be used to predict the manner in which they did the exercise.  The "classe" variable will be used to determine this.  The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information is available in the section on the Weight Lifting Exercise Dataset from the link:
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

We will build a model from the Training set and use the validation set to predict "classe".  The prediction methods that will be used are **Decision Trees, Gradient Boosting and Random Forest**.  The accuracy and out of Sample Error will be calculated and compared for each model.

The prediction model with the best accuracy and hence lowest error will be used to predict "classe" values for the testing set of 20 observations provided in the link.

## Loading the training data

```r
url <- "C:/Users/gwan/Desktop/pml-training.csv"
training = read.csv(url)

## get the columns/rows available
dim(training)
```

```
## [1] 19622   160
```

```r
## Sample Structure of the dataset
str(training[,1:10])
```

```
## 'data.frame':	19622 obs. of  10 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name           : chr  "carlitos" "carlitos" "carlitos" "carlitos" ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp      : chr  "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" ...
##  $ new_window          : chr  "no" "no" "no" "no" ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
```

## Loading the testing data

```r
url <- "C:/Users/gwan/Desktop/pml-testing.csv"
testingfinal = read.csv(url)

## get the rows/columns available
dim(testingfinal)
```

```
## [1]  20 160
```

```r
## Sample Structure of the dataset
str(testingfinal[,1:10])
```

```
## 'data.frame':	20 obs. of  10 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name           : chr  "pedro" "jeremy" "jeremy" "adelmo" ...
##  $ raw_timestamp_part_1: int  1323095002 1322673067 1322673075 1322832789 1322489635 1322673149 1322673128 1322673076 1323084240 1322837822 ...
##  $ raw_timestamp_part_2: int  868349 778725 342967 560311 814776 510661 766645 54671 916313 384285 ...
##  $ cvtd_timestamp      : chr  "05/12/2011 14:23" "30/11/2011 17:11" "30/11/2011 17:11" "02/12/2011 13:33" ...
##  $ new_window          : chr  "no" "no" "no" "no" ...
##  $ num_window          : int  74 431 439 194 235 504 485 440 323 664 ...
##  $ roll_belt           : num  123 1.02 0.87 125 1.35 -5.92 1.2 0.43 0.93 114 ...
##  $ pitch_belt          : num  27 4.87 1.82 -41.6 3.33 1.59 4.44 4.15 6.72 22.4 ...
##  $ yaw_belt            : num  -4.75 -88.9 -88.5 162 -88.6 -87.7 -87.3 -88.5 -93.7 -13.1 ...
```
This finaltesting dataset will be used for out final prediction of "classe".

## Loading the packages 

```r
library(ggplot2)
library(lattice)
library(caret)
```

```
## Warning: package 'caret' was built under R version 4.0.4
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 4.0.5
```

```
## randomForest 4.6-14
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
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 4.0.5
```

```
## Loading required package: tibble
```

```
## Loading required package: bitops
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## 
## Attaching package: 'rattle'
```

```
## The following object is masked from 'package:randomForest':
## 
##     importance
```

```r
## set the seed data
set.seed(300)
```

## Cleaning data

```r
## Remove the first 7 columns which are not relevant to the study
training <- training[,-c(1:7)]

## Remove variables that have little variability
nvz <- nearZeroVar(training)
training <- training[,-nvz]

## Remove variables with NAS
training <- training[,colSums(is.na(training))==0]

dim(training)
```

```
## [1] 19622    53
```
The training dataset have now 53 columns


## Splitting the training set
The training set is divided in two parts one for training and the other for cross validation.
70% is used as training set and 30% as the validation set.


```r
trainInd <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
train <- training[trainInd,]
test  <- training[-trainInd,]
dim(train)
```

```
## [1] 13737    53
```

```r
dim(test)
```

```
## [1] 5885   53
```

```r
## create a train control variable with sample method of cross validation and number of folds = 3
control <- trainControl(method="cv", number=3, verboseIter=FALSE)
```

# Building and Testing the Models

## 1. Decision Tree

```r
fitdt <- train(classe ~ .,data=train,method="rpart")
preddt <- predict(fitdt,test)

# need to covert to factor as test$classe is chr
cfm <- confusionMatrix(preddt, as.factor(test$classe))

cfm$overall[1]
```

```
##  Accuracy 
## 0.4929482
```

```r
fancyRpartPlot(fitdt$finalModel)
```

![](machine_learning_files/figure-html/decisiontree-1.png)<!-- -->

## 2. Gradient Boosting 

```r
fitbt <- train(classe ~ ., method="gbm", data=train, trControl=control, tuneLength = 5,verbose=FALSE)
predbt <- predict(fitbt,test)
# need to convert to factor as test$classe is chr
cfm <- confusionMatrix(predbt, as.factor(test$classe))
cfm$overall[1]
```

```
##  Accuracy 
## 0.9892948
```

```r
plot(fitbt)
```

![](machine_learning_files/figure-html/boost-1.png)<!-- -->


## 3. Random Forest

```r
fitrf <- train(classe~ .,data=train, method="rf", trControl=control, tuneLength = 5)
predrf <- predict(fitrf,test)
# need to covert to factor as test$classe is chr
cfm <- confusionMatrix(predrf, as.factor(test$classe))
cfm$overall[1]
```

```
##  Accuracy 
## 0.9964316
```

```r
plot(fitrf,main="Figure 1: Random Forest : Mean Decrease Accuracy ")
```

![](machine_learning_files/figure-html/randomforest-1.png)<!-- -->

- The model accuracy plot shows that maximum accuracy is achieved between 10-20 predictors.


```r
v <- randomForest(formula = as.factor(classe) ~ . , data = train, ntree=100,mtry=2, importance = TRUE)
varImpPlot(v, main = "Random Forest : Measurement of variable Importance")
```

![](machine_learning_files/figure-html/randomforest2-1.png)<!-- -->

```r
plot(v,main="Figure 2: Random Forest : Error rate v/s No of Trees")
```

![](machine_learning_files/figure-html/randomforest2-2.png)<!-- -->

- Figure 1: The Mean Decrease Accuracy plot shows that "yaw_belt" is the most important variable.
- Figure 2: The Error Rate decreases as the number of trees increases.

## Comparing the Models
- The accuracy for Decision Tree is 0.4929 and Out of Sample Error is 0.5071
- The accuracy for Gradient Boosting is 0.9893 and Out of Sample Error is 0.0107
- The accuracy for Random Forest is 0.9964 and Out of Sample Error is 0.0036
- As the Random Forest has the best accuracy and lowest Out of Sample error among the model fits, we will apply this model of prediction on the Testing set.

## Prediction on Final Testing data using best model 

```r
pred <- predict(fitrf,newdata=testingfinal)
pred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
The above predictions will be used for the project quiz.
