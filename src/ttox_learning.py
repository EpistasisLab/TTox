# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains functions for building, evaluating, and implementing machine learning models


## Module
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


## This function split a feature-response dataset into K folds 
def split_dataset_into_k_folds(data_df, label, N_k, seed_no = 0):
	## 0. Input arguments: 
		# data_df: data frame that contains the learning data 
		# label: name of label(response) column
		# N_k: number of folds to split into 
		# seed_no: seed number to use

	# 1. Separate data into feature and response 
	X, y = data_df.drop(label, axis = 1), data_df[label]

	# 2. Split data into K folds
	kf = KFold(n_splits = N_k, random_state = seed_no, shuffle = True)
	data_list = []
	for train_index, test_index in kf.split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		fold_data = X_train, y_train, X_test, y_test
		data_list.append(fold_data)

	return data_list


## This function learns regression model from training data, then tests the model on testing data and returns model coefficient of determination(R squared)
def evaluate_regression_model_performance(X_train, X_test, y_train, y_test, method, seed_no = 0):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# method: regression method to be used: 'RandomForest', or 'XGBoost'
		# seed_no: seed number to use

	## 1. Define function for regression method
	if method == 'RandomForest':
		regressor = RandomForestRegressor(random_state = seed_no)
	if method == 'XGBoost':
		regressor = xgb.XGBRegressor(random_state = seed_no) 
	
	## 2. Learn model on training data  
	regressor.fit(X_train, y_train)
	
	## 3. Test model on testing data 
	# make predictions using testing feature data 
	y_pred = regressor.predict(X_test)	
	# compare with testing response data, compute r squared 
	y_r2 = r2_score(y_test, y_pred)

	return regressor, y_r2 


## This function learns classification model from training data, then tests the model on testing data and returns model AUROC
def evaluate_classification_model_performance(X_train, X_test, y_train, y_test, method, seed_no = 0):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# method: classification method to be used: 'RandomForest', or 'XGBoost'
		# seed_no: seed number to use

	## 1. Define function for classification method
	if method == 'RandomForest':
		classifier = RandomForestClassifier(random_state = seed_no)
	if method == 'XGBoost':
		classifier = xgb.XGBClassifier(random_state = seed_no)

	## 2. Learn model on training data 
	classifier.fit(X_train, y_train)

	## 3. Test model on testing data 
	# make predictions using testing feature data 
	y_pred_prob = classifier.predict_proba(X_test)[:,1]
	# compare with testing response data, compute r squared 
	y_auc = roc_auc_score(y_test, y_pred_prob) 

	return classifier, y_auc


## This function builds supervised learning model from training data, then tests the model on testing data and returns model performance
def evaluate_model_performance(X_train, X_test, y_train, y_test, task, method, seed_number = 0):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# task: type of supervised learning task: 'regression' or 'classification' 
		# method: regression method to be used: 'RandomForest', or 'XGBoost'
		# seed_number: seed number to use

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
		model, metric = evaluate_regression_model_performance(X_train, X_test, y_train, y_test, method, seed_no = seed_number)
	# classification task
	if task == 'classification':
		model, metric = evaluate_classification_model_performance(X_train, X_test, y_train, y_test, method, seed_no = seed_number)

	return N_feature, N_train, N_test, model, metric


## This function evaluates performance of a supervised learning model by cross-validation
def evaluate_model_performance_by_cv(data_df, features, label, N_repeat, N_k, task, method, seed_start = 0):
	## 0. Input arguments:
		# data_df: data frame that contains the learning data 
		# features: features to be used for model construction 
		# label: name of label(response) column
		# N_repeat: number of independent cross-validation runs, each run will generate one performance score 
		# N_k: number of folds to split data into
		# task: type of supervised learning task: 'regression' or 'classification' 
		# method: regression method to be used: 'RandomForest', or 'XGBoost'
		# seed_start: seed number to be used in the first run, 'seed_start + 1' will be used for the second run, ... 
	
	## 1. Repeat the analysis for 'N_repeat' times
	cv_metric = []
	for i in range(0, N_repeat):
		# split data into 'N_k' folds  
		i_data_split = split_dataset_into_k_folds(data_df, label, N_k, seed_no = seed_start + i)		
		# iterate by folds
		i_cv_metric = []
		for ids in i_data_split:
			# obtain split of the fold 
			ids_feat_train, ids_label_train, ids_feat_test, ids_label_test = ids
			# evaluate model on the testing fold 
			_, _, _, _, ids_metric = evaluate_model_performance(ids_feat_train[features].values, ids_feat_test[features].values, ids_label_train.values, ids_label_test.values, task, method)
			i_cv_metric.append(ids_metric)
		# average performance of all folds 
		cv_metric.append(np.mean(i_cv_metric))

	return cv_metric


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
