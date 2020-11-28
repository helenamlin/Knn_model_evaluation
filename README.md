---
title: "Eval Lab"
author: "Helena Lindsay, Max St John, Allen Baiju"
date: "11/3/2020"
output: html_document
editor_options: 
  chunk_output_type: console
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(readr)
library(dplyr)
library(caret)
library(ROCR)
library(Metrics)
library(MLmetrics)
library(knitr)
```

## CNN Data {.tabset}

### Tree Based

#### Part 1

```{r, warning=FALSE, include=FALSE}
commlabels = read_csv("cnn_commmercial_label.csv", col_names = FALSE)
commlabels = t(commlabels)
CNNdata = read_csv("tv_commercial_datasets_CNN_Cleaned.csv", col_names = commlabels)
CNNdata = CNNdata[-c(1),]
CNNdata$label <- as.factor(CNNdata$label)
```


```{r, include=FALSE}
# Sample 80% of our know data as training and 20% as test
set.seed(1982)
split_index <- createDataPartition(CNNdata$label, p = .8, 
                                  list = FALSE,
                                  times = 1)
train_data <- CNNdata[split_index,]
test <- CNNdata[-split_index,]
CNN_train <- train(label~., 
                   data=train_data, #use the training data
                   method='rpart',# indicates the use of tree based model
                   na.action = na.omit)
```

```{r}
library(RColorBrewer)
coul <- brewer.pal(3, "Set3")
barplot(CNN_train$finalModel$variable.importance, col=coul)
```

```{r, include=FALSE}
CNN_eval <-(predict(CNN_train, newdata = test))
CNN_eval_prob <- predict(CNN_train,newdata = test, type = "prob")
test_outcome = test$label
```

```{r}
confusionMatrix(CNN_eval, test_outcome, positive = "1", dnn=c("Prediction", "Actual"), mode = "sens_spec")
```

```{r}
baserate = table(CNNdata$label)[2] / sum(table(CNNdata$label))
baserate
```

From this confusion matrix, we can see that the Accuracy rate is 76.09%, which is 12.17% higher than the base rate. The Kappa of 47.81% would be classified as a moderate score. We got 82.13% and 65.38% for sensitivity (TPR) and specificity rates respectively.


```{r, echo=FALSE}
adjust_thres <- function(x, y, z) {
  #x=pred_probablities, y=threshold, z=test_outcome
  thres <- as.factor(ifelse(x > y, 1,-1))
  confusionMatrix(thres, z, positive = "1", dnn=c("Prediction", "Actual"), mode = "everything")
}
```

```{r, echo=FALSE}
CNN_eval <- data.frame(pred_class=CNN_eval, pred_prob=CNN_eval_prob$`1`,target=as.numeric(test_outcome))
pred <- prediction(CNN_eval$pred_prob,CNN_eval$target)
tree_perf <- performance(pred,"tpr","fpr")
plot(tree_perf, colorize=TRUE) + abline(a=0, b= 1)
```

```{r}
tree_perf_AUC <- performance(pred,"auc")
print(tree_perf_AUC@y.values)
```
Generally, AUC values between 0.7 and 0.8 are considered acceptable, 0.8 and 0.9 are considered excellent, and over 0.9 are considered outstanding. In our model, the AUC value of 0.7593 shows that the overall accuracy of the model is high enough.


#### Part 2
```{r, echo=FALSE}
CNN_eval$test <- test_outcome
CNN_eval$test <- as.factor(CNN_eval$test)
CNN_eval$pred_prob <- as.numeric(CNN_eval$pred_prob)
plt_cal <- calibration(test ~ pred_prob, data=CNN_eval)
calibration(test ~ pred_prob, data=CNN_eval)
xyplot(plt_cal)
bias <- bias(as.numeric(test_outcome), as.numeric(CNN_eval$pred_class))
test_var <- var(as.numeric(test_outcome))
pred_var <- var(as.numeric(CNN_eval$pred_class))
variance = c(test_var, pred_var)
tab <- tibble(test_var=test_var,
              pred_var=pred_var,
              bias = bias)
kable(tab, format = "markdown")
#If the model was "unbias" the result would be zero. 
CNN_eval$test <- recode(CNN_eval$test, "-1"=0,"1"=1)
```

```{r}
LogLoss3NN <- LogLoss(as.numeric(CNN_eval$pred_prob), as.numeric(CNN_eval$test))
LogLoss3NN
```
Our bias of -0.0106 is close to 0 and thus we can say that there isn't much bias in this model. The variance of our test outcomes was 0.2306 while our prediction variance was 0.2276.
Considering that a good variance score is generally said to be over 60%, we should consider adding more data to improve our model.
Since both the variances and bias are low, we should take into account the possibility of the model being overfit.
The LogLoss value of 0.533 is far from 0, and can be further improved.

#### Part 3
```{r, warning=FALSE, echo=FALSE}
thresholds <- seq(.1, .9, .1)
metrics <- data.frame(sapply(thresholds, function(x) adjust_thres(CNN_eval_prob$`1`,x, test_outcome)$overall))
sensitivity <- sapply(thresholds, function(x) adjust_thres(CNN_eval_prob$`1`,x, test_outcome)$byClass[1])
results <- lapply(thresholds, function(x) adjust_thres(CNN_eval_prob$`1`,x, test_outcome)$table)
FPR.list <- sapply(results, function(x) x[2,1]/(x[2,1]+x[2,2]))
TPR.list <- sapply(results, function(x) x[2,2]/(x[1,2]+x[2,2]))
metrics <- rbind(metrics, sensitivity, FPR.list, TPR.list)
metrics <- metrics[-c(3:7),]
rownames(metrics) <- c("Accuracy", "Kappa", "Sensitivity", "FPR", "TPR")
colnames(metrics) <- thresholds
kable(metrics, format = "markdown")
```
When adjusting the threshold from 0.1 to 0.9, we can see that any threshold between 0.4 and 0.7 would provide us with the most balanced scores across all measures. 


### KNN (3)

#### Part 1
```{r, include=FALSE}
commlabels = read_csv("cnn_commmercial_label.csv", col_names = FALSE)
commlabels = t(commlabels)
CNNdata = read_csv("tv_commercial_datasets_CNN_Cleaned.csv", col_names = commlabels)
CNNdata = CNNdata[-c(1),]
CNNdata$label <- as.numeric(CNNdata$label)
```

```{r}
baserate = table(CNNdata$label)[2] / sum(table(CNNdata$label))
baserate
```
There are 14,411 commercials and 8,134 non-commercials. At random, we have a 63.9% chance of correctly picking out a commercial.


```{r, include=FALSE}
CNNdata2 <- CNNdata %>% select(-contains("var"))
```


```{r, warning=FALSE}
commercial_correlations = cor(CNNdata2)
head(commercial_correlations)
```
Motion distribution is highly correlated with frame differential distribution and motion distribution with values of 0.715 and -0.757, respectively. Short time energy is highly correlated with spectral flux with a value of 0.823. Spectral centroid is highly correlated with spectral roll off with a value of 0.809.

```{r, include=FALSE}
subsetvars = c("shot_length", "zcr_mn", "fundamental_freq_mn", "label")
kdata = CNNdata2[subsetvars]
```



```{r,echo=TRUE, warning=FALSE, include=FALSE}
set.seed(1982)
split_index <- createDataPartition(kdata$label, p = .8, 
                                  list = FALSE,
                                  times = 1)
train_data <- kdata[split_index,]
test <- kdata[-split_index,]
```


We can see from the above plot that shot_length is the largest driver of the outcome and has a significant impact on the outcome values. 

```{r}
set.seed(1982)
library(class)
cl = train_data[, "label"]
commercials_3NN <-  knn(train = train_data,
               test = test,
               cl = train_data$label,
               k = 3,
               prob = TRUE) 
```


```{r,warning=FALSE, include=FALSE}
prob_knn <- tibble(commercials_3NN, attributes(commercials_3NN)$prob)
prob_knn$prob <- if_else(prob_knn$commercials_3NN==-1,1-prob_knn$`attributes(commercials_3NN)$prob`, prob_knn$`attributes(commercials_3NN)$prob`)
test_outcome = as.factor(test$label)
prob_knn$test <- test_outcome
```

```{r,warning=FALSE}
confusionMatrix(commercials_3NN, as.factor(test_outcome), positive = "1", dnn=c("Prediction", "Actual"), mode = "sens_spec")
```
From this confusion matrix, we can see that the Accuracy rate is 96.9%, which is 32.8% higher than the base rate. The Kappa of 93.26% would be classified as a very good score. We got 97.51% and 95.81% for sensitivity (TPR) and specificity rates respectively.Just by looking at the confusion matrix, we should question the validity of the model, since all the measures seem very high.

```{r, include=FALSE}
adjust_thres <- function(x, y, z) {
  thres <- as.factor(ifelse(x > y, 1,-1))
  confusionMatrix(thres, z, positive = "1", dnn=c("Prediction", "Actual"), mode = "everything")
}
```


```{r, include=FALSE}
commercials_3NN <- data.frame(pred_class=commercials_3NN, pred_prob=prob_knn$prob,target=as.numeric(test_outcome))
pred <- prediction(commercials_3NN$pred_prob,commercials_3NN$target)
tree_perf <- performance(pred,"tpr","fpr")
```

```{r,warning=FALSE}
plot(tree_perf, colorize=TRUE) + abline(a=0, b= 1)
```

```{r,warning=FALSE, include=FALSE}
tree_perf_AUC <- performance(pred,"auc")
tree_perf_AUC@y.values
```
As seen in the ROC curve and the AUC value of 0.9873, the model seem to excellent classification accuracy. However, we should note that it is very unlikely for a model to have this high of a classification accuracy and thus we should consider the possibility of this model being over-fitting. 



#### Part 2
```{r, echo=FALSE}
commercials_3NN$pred_prob <- as.numeric(commercials_3NN$pred_prob)
commercials_3NN$test <- test_outcome
plt_cal <- calibration(test ~ pred_prob, data=commercials_3NN)
calibration(test ~ pred_prob, data=commercials_3NN)
xyplot(plt_cal)
bias <- bias(as.numeric(test_outcome), as.numeric(commercials_3NN$pred_class))
bias
#If the model was "unbias" the result would be zero. 
test_var <- var(as.numeric(test_outcome))
pred_var <- var(as.numeric(commercials_3NN$pred_class))
variance = c(test_var, pred_var)
tab <- tibble(test_var=test_var,
              pred_var=pred_var,
              bias = bias)
kable(tab, format = "markdown")
commercials_3NN$test <- recode(commercials_3NN$test, "-1"=0,"1"=1)
```

```{r}
LogLoss3NN <- LogLoss(as.numeric(commercials_3NN$pred_prob), as.numeric(commercials_3NN$test))
LogLoss3NN
```
Our bias of 0.0008871 is close to 0 and thus we can say that there isn't much bias in this model. The variance of our test outcomes was 0.2304 while our prediction variance was 0.2306. Considering that a good variance score is generally said to be over 60%, we should consider adding more data to improve our model.
The LogLoss value of 0.39 is a relatively good score. 


#### Part 3
```{r, echo=FALSE}
thresholds <- seq(.1, .9, .1)
metrics <- data.frame(sapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$overall))
sensitivity <- sapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$byClass[1])
results <- lapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$table)
FPR.list <- sapply(results, function(x) x[2,1]/(x[2,1]+x[2,2]))
TPR.list <- sapply(results, function(x) x[2,2]/(x[1,2]+x[2,2]))
metrics <- rbind(metrics, sensitivity, FPR.list, TPR.list)
metrics <- metrics[-c(3:7),]
rownames(metrics) <- c("Accuracy", "Kappa", "Sensitivity", "FPR", "TPR")
colnames(metrics) <- thresholds
kable(metrics, format = "markdown")
```
When adjusting the threshold from 0.1 to 0.9, we can see that any threshold between 0.4 and 0.6 would provide us with the most balanced scores across all measures. 

### KNN (11)

#### Part 1
```{r, echo=FALSE}
chooseK = function(k, train_set, val_set, train_class, val_class){
  
  # Build knn with k neighbors considered.
  set.seed(1)
  class_knn = knn(train = train_set,    #<- training set cases
                  test = val_set,       #<- test set cases
                  cl = train_class,     #<- category for classification
                  k = k,                #<- number of neighbors considered
                  use.all = TRUE)       #<- control ties between class assignments
                                        #   If true, all distances equal to the kth largest are included
  conf_mat = table(class_knn, val_class)
  
  # Calculate the accuracy.
  accu = sum(conf_mat[row(conf_mat) == col(conf_mat)]) / sum(conf_mat)                         
  cbind(k = k, accuracy = accu)
}
kdata_diffk = sapply(seq(1, 21, by = 2),  #<- set k to be odd number from 1 to 21
                         function(x) chooseK(x, 
                                             train_set = train_data[, c("shot_length", "zcr_mn", "fundamental_freq_mn"), drop = FALSE],
                                             val_set = test[, c("shot_length", "zcr_mn", "fundamental_freq_mn"), drop = FALSE],
                                             train_class = train_data$label,
                                             val_class = test$label))
kdata_diffk = data.frame(k = kdata_diffk[1,],
                             accuracy = kdata_diffk[2,])
```

```{r, echo=FALSE}
ggplot(kdata_diffk,
       aes(x = k, y = accuracy)) +
  geom_line(color = "orange", size = 1.5) +
  geom_point(size = 3)
```

From the above elbow chart, we decided to use k=11 for further analysis.



```{r, include=FALSE}
set.seed(1982)
library(class)
cl = train_data[, "label"]
commercials_11NN <-  knn(train = train_data,
               test = test,
               cl = train_data$label,
               k = 11,
               prob = TRUE) 
```




```{r, echo=FALSE}
prob_knn <- tibble(commercials_11NN, attributes(commercials_11NN)$prob)
prob_knn$prob <- if_else(prob_knn$commercials_11NN==-1,1-prob_knn$`attributes(commercials_11NN)$prob`, prob_knn$`attributes(commercials_11NN)$prob`)
test_outcome = as.factor(test$label)
confusionMatrix(commercials_11NN, as.factor(test_outcome), positive = "1", dnn=c("Prediction", "Actual"), mode = "sens_spec")
adjust_thres <- function(x, y, z) {
  thres <- as.factor(ifelse(x > y, 1,-1))
  confusionMatrix(thres, z, positive = "1", dnn=c("Prediction", "Actual"), mode = "everything")
}
prob_knn$test <- test_outcome
```

The k=11 model has a high accuracy rate of 94.15%, along with a strong kappa value of 87.22%. Sensitivity and specificity rates are both very good as well.


```{r, echo=FALSE}
commercials_11NN <- data.frame(pred_class=commercials_11NN, pred_prob=prob_knn$prob,target=as.numeric(test_outcome))
pred <- prediction(commercials_11NN$pred_prob,commercials_11NN$target)
tree_perf <- performance(pred,"tpr","fpr")
plot(tree_perf, colorize=TRUE) + abline(a=0, b= 1)
```

```{r}
tree_perf_AUC <- performance(pred,"auc")
tree_perf_AUC@y.values
```
As seen in the ROC curve and the AUC value, this model seem to be able to classify with a very high classification accuracy. Although less extreme, similar to our k=3 model, this model should also be treated with regards to the possibility of over-fitting. 

#### Part 2
```{r, echo=FALSE}
commercials_11NN$pred_prob <- as.numeric(commercials_11NN$pred_prob)
commercials_11NN$test <- test_outcome
plt_cal <- calibration(test ~ pred_prob, data=commercials_11NN)
calibration(test ~ pred_prob, data=commercials_11NN)
xyplot(plt_cal)
commercials_11NN$pred_class <- recode(commercials_11NN$pred_class, "-1"=0,"1"=1)
bias <- bias(as.numeric(test_outcome), as.numeric(commercials_11NN$pred_class))
bias
#If the model was "unbias" the result would be zero. 
test_var <- var(as.numeric(test_outcome))
pred_var <- var(as.numeric(commercials_11NN$pred_class))
variance = c(test_var, pred_var)
tab <- tibble(test_var=test_var,
              pred_var=pred_var,
              bias = bias)
kable(tab, format = "markdown")
```

```{r}
commercials_11NN$test <- recode(commercials_11NN$test, "-1"=0,"1"=1)
LogLoss11NN <- LogLoss(as.numeric(commercials_11NN$pred_prob), as.numeric(commercials_11NN$test))
LogLoss11NN
```
Our bias of 0.99 can be closer to 0 and thus we believe that there is potential for this model to improve. The variance of our test outcomes was 0.23 while our prediction variance was 0.2277. Considering that a good variance score is generally said to be over 60%, we should consider adding more data to improve our model.
The LogLoss value of 0.256 is a relatively good score. 


#### Part 3
```{r, echo=FALSE}
thresholds <- seq(.1, .9, .1)
metrics <- data.frame(sapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$overall))
sensitivity <- sapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$byClass[1])
results <- lapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$table)
FPR.list <- sapply(results, function(x) x[2,1]/(x[2,1]+x[2,2]))
TPR.list <- sapply(results, function(x) x[2,2]/(x[1,2]+x[2,2]))
metrics <- rbind(metrics, sensitivity, FPR.list, TPR.list)
metrics <- metrics[-c(3:7),]
rownames(metrics) <- c("Accuracy", "Kappa", "Sensitivity", "FPR", "TPR")
colnames(metrics) <- thresholds
kable(metrics, format = "markdown")
```
When adjusting the threshold from 0.1 to 0.9, we can see that any threshold between 0.5 and 0.6 would provide us with the most balanced scores across all measures. 

## Loan Data {.tabset}

### Tree Based
#### Part 1

```{r, echo=FALSE}
library(tidyverse)
library(caret)
bank_data = read.csv("bank.csv", 
                     check.names = FALSE,
                     stringsAsFactors = FALSE)
bank_data[bank_data=="unknown"] <- NA
bank_data$age <- as.numeric(bank_data$age)
bank_data$duration <- as.numeric(bank_data$duration)
bank_data$pdays <- as.numeric(bank_data$pdays)
bank_data$previous <- as.numeric(bank_data$previous)
bank_data$education <- recode(bank_data$education, 'tertiary' = 't', 'secondary' = 's', 'primary' = 'p')
bank_data$marital <- recode(bank_data$marital, 'married' = 'm', 'divorced' = 'd', 'single' = 's')
bank_data$`signed up` <- as.factor(bank_data$`signed up`)
bank_data <- bank_data[complete.cases(bank_data), ]
# Sample 80% of our know data as training and 20% as test
set.seed(1982)
split_index <- createDataPartition(bank_data$`signed up`, p = .8, 
                                  list = FALSE,
                                  times = 1)
train_data <- bank_data[split_index,]
test <- bank_data[-split_index,]
bank_train <- train(`signed up`~., 
                   data=train_data, #use the training data
                   method='rpart',# indicates the use of tree based model
                   na.action = na.omit)
```




```{r, echo=FALSE}
library(RColorBrewer)
coul <- brewer.pal(5, "Set3")
barplot(bank_train$finalModel$variable.importance, col=coul)
```

```{r, echo=FALSE}
bank_eval <-(predict(bank_train, newdata = test))
bank_eval_prob <- predict(bank_train,newdata = test, type = "prob")
test_outcome = test$`signed up`
confusionMatrix(bank_eval, test_outcome, positive = "1", dnn=c("Prediction", "Actual"), mode = "sens_spec")
adjust_thres <- function(x, y, z) {
  #x=pred_probablities, y=threshold, z=test_outcome
  thres <- as.factor(ifelse(x > y, 1,0))
  confusionMatrix(thres, z, positive = "1", dnn=c("Prediction", "Actual"), mode = "everything")
}
```
From this confusion matrix, we can see that the Accuracy rate is 83.9%. The Kappa of 46.29% would be classified as a moderate score. The sensitivity (TPR) rate of 44.54% isn't as high as we wanted it to be, but the specificity(TNR) rate of 95.3% is very high. 


```{r, echo=FALSE}
library(ROCR)
bank_eval <- data.frame(pred_class=bank_eval, pred_prob=bank_eval_prob$`1`,target=as.numeric(test_outcome))
pred <- prediction(bank_eval$pred_prob,bank_eval$target)
tree_perf <- performance(pred,"tpr","fpr")
plot(tree_perf, colorize=TRUE) + abline(a=0, b= 1)
```

```{r, echo=FALSE}
tree_perf_AUC <- performance(pred,"auc")
print(tree_perf_AUC@y.values)
```
The AUC score of 0.715 would be considered a fair score, and we can most likely eliminate the possibility of the model being over-fitting or under-fitting in this case. 


#### Part 2

```{r, echo=FALSE}
bank_eval$test <- test_outcome
plt_cal <- calibration(test ~ pred_prob, data=bank_eval)
calibration(test ~ pred_prob, data=bank_eval)
library(Metrics)
bias <- bias(as.numeric(bank_eval$test), as.numeric(bank_eval$pred_class))
bias
#install.packages("MLmetrics")
library(MLmetrics)
test_var <- var(as.numeric(test_outcome))
pred_var <- var(as.numeric(bank_eval$pred_class))
variance = c(test_var, pred_var)
tab <- tibble(test_var=test_var,
              pred_var=pred_var,
              bias = bias)
kable(tab, format = "markdown")
```

```{r}
LogLoss_bank <- LogLoss(as.numeric(bank_eval$pred_prob), as.numeric(bank_eval$test))
LogLoss_bank
```
The bias score of 0.088 is low enough to consider that this model is unbiased, but this model also suggests a low variance in both the test outcomes and predicted outcomes. The low variances suggest the possibility of over-fitting of this model.
Our LogLoss of 1.845 is very high, and can be improved as well.

#### Part 3
```{r, warning=FALSE}
thresholds <- seq(.1, .9, .1)
metrics <- data.frame(sapply(thresholds, function(x) adjust_thres(bank_eval_prob$`1`,x, test_outcome)$overall))
sensitivity <- sapply(thresholds, function(x) adjust_thres(bank_eval_prob$`1`,x, test_outcome)$byClass[1])
results <- lapply(thresholds, function(x) adjust_thres(bank_eval_prob$`1`,x, test_outcome)$table)
FPR.list <- sapply(results, function(x) x[2,1]/(x[2,1]+x[2,2]))
TPR.list <- sapply(results, function(x) x[2,2]/(x[1,2]+x[2,2]))
metrics <- rbind(metrics, sensitivity, FPR.list, TPR.list)
metrics <- metrics[-c(3:7),]
rownames(metrics) <- c("Accuracy", "Kappa", "Sensitivity", "FPR", "TPR")
colnames(metrics) <- thresholds
kable(metrics, format = "markdown")
```
When adjusting the threshold from 0.1 to 0.9, we can see that any threshold between 0.2 and 0.3 would provide us with the most balanced scores across all measures.


### KNN (3)
#### Part 1
```{r, include=FALSE}
bank_data = read.csv("bank.csv", 
                     check.names = FALSE,
                     stringsAsFactors = FALSE)
bank_data[, c("age", "duration", "balance")] <- lapply(bank_data[, c("age", "duration", "balance")], function(x) scale(x))
table(bank_data$`signed up`)[2]/ sum(table(bank_data$`signed up`))
# Sample 80% of our know data as training and 20% as test
set.seed(1982)
bank_data_train_rows = sample(1:nrow(bank_data), round(0.8*nrow(bank_data)), replace = FALSE)
bank_data_train = bank_data[bank_data_train_rows, ]
bank_data_test = bank_data[-bank_data_train_rows, ]
bank_3NN <- knn(train = bank_data_train[, c("age", "duration", "balance")],
                test = bank_data_test[, c("age", "duration", "balance")], 
                cl = bank_data_train[,"signed up"],
                k = 3,
                use.all = TRUE,
                prob = TRUE)
```



```{r, include=FALSE}
prob_knn <- tibble(bank_3NN, attributes(bank_3NN)$prob)
prob_knn$prob <- if_else(prob_knn$bank_3NN==0,1-prob_knn$`attributes(bank_3NN)$prob`, prob_knn$`attributes(bank_3NN)$prob`)
test_outcome = as.factor(bank_data_test$`signed up`)
prob_knn$test <- test_outcome
```

```{r}
confusionMatrix(bank_3NN, as.factor(test_outcome), positive = "1", dnn=c("Prediction", "Actual"), mode = "sens_spec")
```

This model has a good accuracy rate of 86.45%, but has a very low kappa score of 0.2297. It also has a low sensitivity rate of 24.565% with a high specificity rate of 94.77%. 

```{r, include=FALSE}
adjust_thres <- function(x, y, z) {
  thres <- as.factor(ifelse(x > y, 1,0))
  confusionMatrix(thres, z, positive = "1", dnn=c("Prediction", "Actual"), mode = "everything")
}
```


```{r, include=FALSE}
bank_3NN <- data.frame(pred_class=bank_3NN, pred_prob=prob_knn$prob,target=as.numeric(test_outcome))
pred <- prediction(bank_3NN$pred_prob,bank_3NN$target)
tree_perf <- performance(pred,"tpr","fpr")
```


```{r}
plot(tree_perf, colorize=TRUE) + abline(a=0, b= 1)
```

```{r}
tree_perf_AUC <- performance(pred,"auc")
tree_perf_AUC@y.values
```
The AUC of 0.6926 isn't too good, and suggests that the model doesn't perform well in classifying data. 

#### Part 2
```{r, echo=FALSE}
bank_3NN$test <- test_outcome
plt_cal <- calibration(test ~ pred_prob, data=bank_3NN)
calibration(test ~ pred_prob, data=bank_3NN)
xyplot(plt_cal)
bias <- bias(as.numeric(test_outcome), as.numeric(bank_3NN$pred_class))
bias
#If the model was "unbias" the result would be zero. 
test_var <- var(as.numeric(test_outcome))
pred_var <- var(as.numeric(bank_3NN$pred_class))
variance = c(test_var, pred_var)
tab <- tibble(test_var=test_var,
              pred_var=pred_var,
              bias = bias)
kable(tab, format = "markdown")
```

```{r}
LogLoss3NN <- LogLoss(as.numeric(bank_3NN$pred_prob), as.numeric(bank_3NN$test))
LogLoss3NN
```
Although this model has a low bias score, the variance for the test outcomes and prediction outcomes are very low, and thus suggests that this model may have a high bias.
Our LogLoss was overwhelmingly high.

#### Part 3
```{r, echo=FALSE}
thresholds <- seq(.1, .9, .1)
metrics <- data.frame(sapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$overall))
sensitivity <- sapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$byClass[1])
results <- lapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$table)
FPR.list <- sapply(results, function(x) x[2,1]/(x[2,1]+x[2,2]))
TPR.list <- sapply(results, function(x) x[2,2]/(x[1,2]+x[2,2]))
metrics <- rbind(metrics, sensitivity, FPR.list, TPR.list)
metrics <- metrics[-c(3:7),]
rownames(metrics) <- c("Accuracy", "Kappa", "Sensitivity", "FPR", "TPR")
colnames(metrics) <- thresholds
kable(metrics, format = "markdown")
```
When looking at the different thresholds and the outcomes for different measures, we can see that a threshold between 0.4 and 0.6 would allow the model to perform at it's maximum capacity.

### KNN (11)
#### Part 1
```{r, echo=FALSE}
chooseK = function(k, train_set, val_set, train_class, val_class){
  
  # Build knn with k neighbors considered.
  set.seed(1)
  class_knn = knn(train = train_set,    #<- training set cases
                  test = val_set,       #<- test set cases
                  cl = train_class,     #<- category for classification
                  k = k,                #<- number of neighbors considered
                  use.all = TRUE)       #<- control ties between class assignments
                                        #   If true, all distances equal to the kth largest are included
  conf_mat = table(class_knn, val_class)
  
  # Calculate the accuracy.
  accu = sum(conf_mat[row(conf_mat) == col(conf_mat)]) / sum(conf_mat)                         
  cbind(k = k, accuracy = accu)
}
kdata_diffk = sapply(seq(1, 21, by = 2),  #<- set k to be odd number from 1 to 21
                         function(x) chooseK(x, 
                                             train_set = bank_data_train[, c("age", "duration", "balance")],
                                             val_set = bank_data_test[, c("age", "duration", "balance")],
                                             train_class = bank_data_train[,"signed up"],
                                             val_class = bank_data_test[,"signed up"]))
kdata_diffk = data.frame(k = kdata_diffk[1,],
                             accuracy = kdata_diffk[2,])
```

```{r, echo=FALSE}
ggplot(kdata_diffk,
       aes(x = k, y = accuracy)) +
  geom_line(color = "orange", size = 1.5) +
  geom_point(size = 3)
```

From the elbow chart, we decided to use k=11 for further analysis.


```{r}
bank_data_train_rows = sample(1:nrow(bank_data), round(0.8*nrow(bank_data)), replace = FALSE)
bank_data_train = bank_data[bank_data_train_rows, ]
bank_data_test = bank_data[-bank_data_train_rows, ]
bank_11NN <- knn(train = bank_data_train[, c("age", "duration", "balance")],
                test = bank_data_test[, c("age", "duration", "balance")], 
                cl = bank_data_train[,"signed up"],
                k = 11,
                use.all = TRUE,
                prob = TRUE)
```



```{r}
prob_knn <- tibble(bank_11NN, attributes(bank_11NN)$prob)
prob_knn$prob <- if_else(prob_knn$bank_11NN==0,1-prob_knn$`attributes(bank_11NN)$prob`, prob_knn$`attributes(bank_11NN)$prob`)
test_outcome = as.factor(bank_data_test$`signed up`)
prob_knn$test <- test_outcome
```

```{r}
confusionMatrix(bank_11NN, as.factor(test_outcome), positive = "1", dnn=c("Prediction", "Actual"), mode = "sens_spec")
```
Similar to our k=3 model, this model also has a solid accuracy rate but with a low kappa score of 0.2516. The specificity rate is higher than the sensitivity rate for this model as well. The balanced accuracy rate of 59.37% isn't as high as we hoped it to be.  

```{r, include=FALSE}
adjust_thres <- function(x, y, z) {
  thres <- as.factor(ifelse(x > y, 1,0))
  confusionMatrix(thres, z, positive = "1", dnn=c("Prediction", "Actual"), mode = "everything")
}
```


```{r, include=FALSE}
bank_11NN <- data.frame(pred_class=bank_11NN, pred_prob=prob_knn$prob,target=as.numeric(test_outcome))
pred <- prediction(bank_11NN$pred_prob,bank_11NN$target)
tree_perf <- performance(pred,"tpr","fpr")
```


```{r}
plot(tree_perf, colorize=TRUE) + abline(a=0, b= 1)
```

```{r}
tree_perf_AUC <- performance(pred,"auc")
tree_perf_AUC@y.values
```
The AUC value for this model is not bad with a 0.7975 score.

#### Part 2
```{r, echo=FALSE}
bank_11NN$pred_prob <- as.numeric(bank_11NN$pred_prob)
bank_11NN$test <- test_outcome
plt_cal <- calibration(test ~ pred_prob, data=bank_11NN)
calibration(test ~ pred_prob, data=bank_11NN)
xyplot(plt_cal)
bias <- bias(as.numeric(test_outcome), as.numeric(bank_11NN$pred_class))
bias
#If the model was "unbias" the result would be zero. 
test_var <- var(as.numeric(test_outcome))
pred_var <- var(as.numeric(bank_11NN$pred_class))
variance = c(test_var, pred_var)
tab <- tibble(test_var=test_var,
              pred_var=pred_var,
              bias = bias)
kable(tab, format = "markdown")
```


```{r}
LogLoss11NN <- LogLoss(as.numeric(bank_11NN$pred_prob), as.numeric(bank_11NN$test))
LogLoss11NN
```
This model also has a low bias score but low variance scores for both test outcomes and prediction outcomes. This suggests the model being highly biased.
The LogLoss score of 18.54 also have room for improvement.

#### Part 3
```{r, echo=FALSE}
thresholds <- seq(.1, .9, .1)
metrics <- data.frame(sapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$overall))
sensitivity <- sapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$byClass[1])
results <- lapply(thresholds, function(x) adjust_thres(prob_knn$prob,x, test_outcome)$table)
FPR.list <- sapply(results, function(x) x[2,1]/(x[2,1]+x[2,2]))
TPR.list <- sapply(results, function(x) x[2,2]/(x[1,2]+x[2,2]))
metrics <- rbind(metrics, sensitivity, FPR.list, TPR.list)
metrics <- metrics[-c(3:7),]
rownames(metrics) <- c("Accuracy", "Kappa", "Sensitivity", "FPR", "TPR")
colnames(metrics) <- thresholds
kable(metrics, format = "markdown")
```
A threshold between 0.2 and 0.3 would allow this model to perform at it's maximum capacity.



# Conclusion

Both models would improve from collecting more data. Increasing both the size of the data set as well as expanding the number of variables used to predict would help improve our performance. 
For our CNN models, KNN=3 performed best with high accuracy, kappa rates on top of low bias and relatively higher variances, which suggests a lower chance of over-fitting.
For our Loan data models, all three models had low variance rates with higher bias scores, which suggested that they may be over-fit. Among the three models, the tree-based model performed the best with the best balance between accuracy, kappa, Sensitivity, and FPR/TPR, although needing more data to further train the model.
Many of our models had mediocre TPR rates, of which we would be able to increase them by lowering the threshold, but this would lead to a higher false positive rate. Banks may be willing to burden a small increase in false positive rate for a larger increase in true positive rate, so this model may be preferred. Our KNN models for the CNN data did an extremely poor job with its predictions. An AUC of almost 1 is a huge red flag, so better data collection is needed to improve this model. 

