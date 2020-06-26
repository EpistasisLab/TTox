# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains functions for building, evaluating, and implementing machine learning models


## Module
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


## This function learns regression model from training data, then tests the model on testing data and returns model coefficient of determination(R squared)
def evaluate_regression_model_performance(X_train, X_test, y_train, y_test, method):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# method: regression method to be used: 'RandomForest', or 'XGBoost'

	## 1. Define function for regression method
	if method == 'RandomForest':
		regressor = RandomForestRegressor(random_state = 0)
	if method == 'XGBoost':
		regressor = xgb.XGBRegressor(random_state = 0) 
	
	## 2. Learn model on training data  
	regressor.fit(X_train, y_train)
	
	## 3. Test model on testing data 
	# make predictions using testing feature data 
	y_pred = regressor.predict(X_test)	
	# compare with testing response data, compute r squared 
	y_r2 = r2_score(y_test, y_pred)

	return regressor, y_r2 


## This function learns classification model from training data, then tests the model on testing data and returns model AUROC
def evaluate_classification_model_performance(X_train, X_test, y_train, y_test, method):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# method: classification method to be used: 'RandomForest', or 'XGBoost'

	## 1. Define function for classification method
	if method == 'RandomForest':
		classifier = RandomForestClassifier(random_state = 0)
	if method == 'XGBoost':
		classifier = xgb.XGBClassifier(random_state = 0)

	## 2. Learn model on training data 
	classifier.fit(X_train, y_train)

	## 3. Test model on testing data 
	# make predictions using testing feature data 
	y_pred_prob = classifier.predict_proba(X_test)[:,1]
	# compare with testing response data, compute r squared 
	y_auc = roc_auc_score(y_test, y_pred_prob) 

	return classifier, y_auc


## This function builds supervised learning model from training data, then tests the model on testing data and returns model performance
def evaluate_model_performance(X_train, X_test, y_train, y_test, task, method):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# task: type of supervised learning task: 'regression' or 'classification' 
		# method: regression method to be used: 'RandomForest', or 'XGBoost'

	## 1. Obtain basic staistics of data 
	# number of features 
	N_feature = X_train.shape[1]
	# number of training instances 
	N_train = X_train.shape[0]
	# number of testing instances 
	N_test = X_test.shape[0]

	## 2. Choose learning function to implement based on the task  
	# regression task 
	if task == 'regression':
		model, metric = evaluate_regression_model_performance(X_train, X_test, y_train, y_test, method)
	# classification task
	if task == 'classification':
		model, metric = evaluate_classification_model_performance(X_train, X_test, y_train, y_test, method)

	return N_feature, N_train, N_test, model, metric


## This function implements supervised learning model to predict response for new instances
def implement_prediciton_model(model, X_pred, task):
	## 0. Input argument: 
		# model: trained supervised learning model  
		# X_pred: array that contains feature data for prediction
		# task: type of supervised learning task: 'regression' or 'classification' 

	## 1. Choose learning function to implement based on whether the task  
	# regression task 
	if task == 'regression':
		y_pred = model.predict(X_pred)
	# classification task
	if task == 'classification':
		y_pred = model.predict_proba(X_pred)[:,1]
	
	return y_pred
