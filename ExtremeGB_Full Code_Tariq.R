library(lattice)
library(ParamHelpers)
library(grid)
library(DMwR)
library(xgboost)
library(mlr)
library(dplyr)
library(tidyverse)
library(MlBayesOpt)
library(Matrix)
library(rBayesianOptimization)


train_data<-read.csv("train_LZdllcl.csv")
test_data<-read.csv("test_2umaH9m.csv")

#Summary of the data
summarizeColumns(train_data)
summarizeColumns(test_data)


#Description of variables
str(train_data)
str(test_data)

#Identify Missing Entires
sapply(train_data, function(x)sum(is.na(x)))

#Identify missing entries in test data
sapply(test_data, function(x)sum(is.na(x)))

#Replace missing values
train_data$previous_year_rating[is.na(train_data$previous_year_rating)]<-0
test_data$previous_year_rating[is.na(test_data$previous_year_rating)]<-0

#Convert previous year's rating, KPI's_met, and award won to categorical variable
#train_data$previous_year_rating<-factor(train_data$previous_year_rating,levels = c(1,2,3,4,5,99),labels = c(1,2,3,4,5,99),exclude = "NA")
#test_data$previous_year_rating<-factor(test_data$previous_year_rating,levels = c(1,2,3,4,5,99),labels = c(1,2,3,4,5,99),exclude = "NA")


#Correct some column names
colnames(train_data)[11]<-"KPI_Score"
colnames(train_data)[12]<-"award_won"
colnames(test_data)[11]<-"KPI_Score"
colnames(test_data)[12]<-"award_won"



#Check imbalance in data
table(train_data$is_promoted)






#Parameter Optimization for Xgboost
#Create dummary variables from the train_dataset
dummy_d<-model.matrix(~department+0,data=train_data)
dummy_r<-model.matrix(~region+0,data=train_data)
dummy_e<-model.matrix(~education+0,data=train_data)
dummy_g<-model.matrix(~gender+0,data=train_data)
dummy_rc<-model.matrix(~recruitment_channel+0,data=train_data)
new_train<-data.frame(train_data[,-c(1,2,3,4,5,6,14)],dummy_d,dummy_r,dummy_e,dummy_g,dummy_rc)

final_data<-cbind(new_train,train_data[,14])
colnames(final_data)[60]<-"is_promoted"


#Using Bayesian Optimization and creating a function
cv_folds <- KFold(final_data$is_promoted, nfolds = 5,stratified = TRUE, seed = 0)

xgb_cv_bayes <- function(eta,gamma,colsample_bytree,max_delta_step,lambda,alpha,
                         max_depth, min_child_weight, subsample) {
  cv <- xgb.cv(params = list(booster = "gbtree",
                             eta = eta,
                             max_depth = max_depth,
                             min_child_weight = min_child_weight,
                             subsample = subsample, 
                             colsample_bytree = colsample_bytree,
                             lambda = lambda,
                             alpha = alpha,
                             gamma=gamma,
                             max_delta_step=max_delta_step,
                             objective = "binary:logistic",
                             eval_metric = "error"),
               data = train_matrix, nrounds=105,folds = cv_folds, prediction = TRUE, 
               showsd = TRUE,early_stopping_rounds = 5, maximize = TRUE, verbose = 0)
  list(Score = cv$evaluation_log$test_error_mean[cv$best_iteration],
       Pred = cv$pred)
}

OPT_Res <- BayesianOptimization(xgb_cv_bayes,
                                bounds = list(max_depth = c(0L,50L),
                                              min_child_weight = c(0,50),
                                              subsample = c(0, 1.0),
                                              eta=c(0,1.0),
                                              colsample_bytree = c(0,1.0),
                                              lambda = c(0,1.0),
                                              alpha = c(0,1.0),
                                              gamma=c(0,50),
                                              max_delta_step=c(0,50)),
                                init_grid_dt = NULL, init_points = 10, n_iter = 60,
                                acq = "ucb", kappa = 2.576, eps = 0.0,verbose = TRUE)


#Using MLBayesOPt Package
res0 <- xgb_cv_opt(data = final_data,
                   label = is_promoted,
                   objectfun = "binary:logistic",
                   evalmetric = "error",
                   n_folds = 5,
                   classes = numberOfClasses,
                   acq = "ucb",
                   init_points = 10,
                   n_iter = 20)


#XGBoost Model Building
X_train<-as.matrix(new_train)
Y_train<-train_data[,14]
train_matrix<-xgb.DMatrix(data=X_train, label=Y_train)
numberOfClasses <- length(unique(train_data$is_promoted))
xgb_params <- list(booster="gbtree",
                   objective = "binary:logistic",
#                   eval_metric = "error",
#                    eta=0.35,
                    subsample=0.8269)
#                   max_depth=4,
#                   alpha=0.3583948,
#                   lambda=0.8668652,
#                   gamma=34.9535110,
#                   min_child_weight=16.1055437,
#                   max_delta_step = 42.8950288,
#                    colsample_bytree=0.9210)


bst<-xgboost(params = xgb_params,data=X_train,label =Y_train,nrounds = 105.3588)
xgb.importance(feature_names = colnames(X_train), bst) %>% xgb.plot.importance()
xgb.plot.tree(model = bst)


#Prediction using model
#training data preparation
dummy_dt<-model.matrix(~department+0,data=test_data)
dummy_rt<-model.matrix(~region+0,data=test_data)
dummy_et<-model.matrix(~education+0,data=test_data)
dummy_gt<-model.matrix(~gender+0,data=test_data)
dummy_rct<-model.matrix(~recruitment_channel+0,data=test_data)

new_test<-data.frame(test_data[,-c(1,2,3,4,5,6)],dummy_dt,dummy_rt,dummy_et,dummy_gt,dummy_rct)


X_test<-as.matrix(new_test)
test_predict<-predict(bst,X_test)


#Write in the submission file
pred<-0
for(i in 1:length(test_predict)){
  if(test_predict[i]<.5){
    pred[i]=0
  }
  else{
    pred[i]=1
  }
}

submission_data<-data.frame(cbind(test_data$employee_id,pred))
colnames(submission_data)<-c("employee_id","is_promoted")
write.table(submission_data,"submission_Tariq.csv",col.names = TRUE,sep = ",",row.names = FALSE)



#F1 Score Formula
f1score_eval <- function(preds, dtrain) {
  e_TP <- sum( (dtrain==1) & (preds >= 0.5) )
  e_FP <- sum( (dtrain==0) & (preds >= 0.5) )
  e_FN <- sum( (dtrain==1) & (preds < 0.5) )
  e_TN <- sum( (dtrain==0) & (preds < 0.5) )
  
  e_precision <- e_TP / (e_TP+e_FP)
  e_recall <- e_TP / (e_TP+e_FN)
  
  e_f1 <- 2*(e_precision*e_recall)/(e_precision+e_recall)
  
  return(list(metric = "f1-score", value = e_f1))
}

#Check F1 Score
train_predict<-predict(bst,X_train)
dtrain<-train_data[,14]
f1score_eval(train_predict,dtrain)


