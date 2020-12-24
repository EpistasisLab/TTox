# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains functions for building, evaluating, and implementing machine learning models


## Module
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

## This function split a feature-response dataset into K folds 
def split_dataset_into_k_folds(data_df, label, N_k, task, seed_no = 0):
	## 0. Input arguments: 
		# data_df: data frame that contains the learning data 
		# label: name of label(response) column
		# N_k: number of folds to split into 
		# task: type of supervised learning task: 'regression' or 'classification'  
		# seed_no: seed number to use

	# 1. Separate data into feature and response 
	X, y = data_df.drop(label, axis = 1), data_df[label]

	# 2. Split data into K folds according to the specified task 
	if task == 'regression':
		kf = KFold(n_splits = N_k, random_state = seed_no, shuffle = True)
		split = kf.split(X)
	# straitified split for classification tasks
	if task == 'classification':
		skf = StratifiedKFold(n_splits = N_k, random_state = seed_no, shuffle = True)
		split = skf.split(X, y)
	# train/test split 
	data_list = []
	for train_index, test_index in split:
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		fold_data = X_train, y_train, X_test, y_test
		data_list.append(fold_data)

	return data_list


## This function computes confidence interval of metric by bootstrapping.
def compute_metric_ci_by_bootsrap(label_vec, pred_vec, confidence_interval = 0.95, bootstrap_times = 1000):
	## 0. Input arguments: 
		# label_vec: input true label arracy
		# pred_vec: input prediction probability array
		# confidence_interval: confidence interval to be computed (number between 0 and 1)
		# bootstrap_times: repeated sampling times for bootstrap

	## 1. Compute confidence interval of mean by bootstrapping
	vec_len = len(pred_vec)
	id_vec = np.arange(0, vec_len)
	# Repeat boostrap process
	sample_props = []
	sample_metrics = []
	np.random.seed(0)
	for sample in range(0, bootstrap_times):
		# Sampling with replacement from the input array
		sample_ids = np.random.choice(id_vec, size = vec_len, replace = True)
		sample_ids = np.unique(sample_ids)
		sample_prop = len(sample_ids)/vec_len
		sample_props.append(sample_prop)
		# compute sample metric
		sample_metric = roc_auc_score(label_vec[sample_ids], pred_vec[sample_ids])
		sample_metrics.append(sample_metric)
	# sort means of bootstrap samples 
	sample_metrics = np.sort(sample_metrics)
	prop = np.mean(sample_props)
	# obtain upper and lower index of confidence interval 
	lower_id = int((0.5 - confidence_interval/2) * bootstrap_times) - 1
	upper_id = int((0.5 + confidence_interval/2) * bootstrap_times) - 1
	ci = (sample_metrics[upper_id] - sample_metrics[lower_id])/2

	return ci, prop


## This function learns classification model from training data, then implements the model on testing data and returns model AUROC and its confidence interval 
def compute_classification_model_auc_ci(X_train, X_test, y_train, y_test, method, n_repeat = 20):
	## 0. Input argument:
		# X_train: array that contains training feature data
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# method: classification method to be used: 'RandomForest', 'XGBoost' or 'Logistic'
		# seed_no: seed number to use

	## 1. Build classfication model from training data, then make predictions on testing data 
	pred_prob_vec = []
	# repeat the analysis multiple times
	for nr in range(0, n_repeat):
		# define function for classification method
		if method == 'RandomForest':
			classifier = RandomForestClassifier(random_state = nr)
		if method == 'XGBoost':
			classifier = xgb.XGBClassifier(random_state = nr)
		if method == 'Logistic':
			classifier = LogisticRegression(penalty = 'none',  random_state = nr)
		# learn model on training data 
		classifier.fit(X_train, y_train)
		# make predictions using testing feature data 
		y_pred_prob = classifier.predict_proba(X_test)[:,1]
		pred_prob_vec.append(y_pred_prob)	
	
	## 2. Take the average prediction from all training models, then compute model AUC and confidence interval    
	y_pred_prob_avg = pd.DataFrame(pred_prob_vec).mean(axis = 0).values
	y_auc = roc_auc_score(y_test, y_pred_prob_avg)
	y_auc_ci, y_auc_prop = compute_metric_ci_by_bootsrap(y_test, y_pred_prob_avg) 
	roc_fpr, roc_tpr, _ = roc_curve(y_test, y_pred_prob_avg)
	
	return y_auc, y_auc_ci, y_auc_prop, roc_fpr, roc_tpr


## This function learns regression model from training data, then implements the model on testing data and returns model coefficient of determination(R squared)
def evaluate_regression_model_performance(X_train, X_test, y_train, y_test, method, seed_no = 0):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# method: regression method to be used: 'RandomForest', 'XGBoost', or 'Linear'
		# seed_no: seed number to use

	## 1. Define function for regression method
	if method == 'RandomForest':
		regressor = RandomForestRegressor(random_state = seed_no)
	if method == 'XGBoost':
		regressor = xgb.XGBRegressor(random_state = seed_no) 
	if method == 'Linear':
		regressor = LinearRegression()

	## 2. Learn model on training data  
	regressor.fit(X_train, y_train)
	
	## 3. Test model on testing data 
	# make predictions using testing feature data 
	y_pred = regressor.predict(X_test)	
	# compare with testing response data, compute metrics 
	y_r2 = r2_score(y_test, y_pred)
	y_mse = mean_squared_error(y_test, y_pred)
	metric_dict = {'r2': y_r2, 'mse': y_mse}

	return regressor, metric_dict


## This function learns classification model from training data, then implements the model on testing data and returns model AUROC
def evaluate_classification_model_performance(X_train, X_test, y_train, y_test, method, seed_no = 0):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# method: classification method to be used: 'RandomForest', 'XGBoost' or 'Logistic'
		# seed_no: seed number to use

	## 1. Define function for classification method
	if method == 'RandomForest':
		classifier = RandomForestClassifier(random_state = seed_no)
	if method == 'XGBoost':
		classifier = xgb.XGBClassifier(random_state = seed_no)
	if method == 'Logistic':
		classifier = LogisticRegression(penalty = 'none',  random_state = seed_no) 		

	## 2. Learn model on training data 
	classifier.fit(X_train, y_train)

	## 3. Test model on testing data 
	# make predictions using testing feature data 
	y_pred_prob = classifier.predict_proba(X_test)[:,1]
	y_pred = classifier.predict(X_test)
	# compare with testing response data, compute metrics
	y_auc = roc_auc_score(y_test, y_pred_prob) 
	y_bac = balanced_accuracy_score(y_test, y_pred)
	y_f1 = f1_score(y_test, y_pred)
	metric_dict = {'auc': y_auc, 'bac': y_bac, 'f1': y_f1}
	
	return classifier, metric_dict


## This function builds supervised learning model from training data, then implements the model on testing data and returns model performance
def evaluate_model_performance(X_train, X_test, y_train, y_test, task, method, seed_number = 0):
	## 0. Input argument:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data 
		# y_test: array that contains testing response data 
		# task: type of supervised learning task: 'regression' or 'classification' 
		# method: regression method to be used: 'RandomForest', 'XGBoost', or 'Linear'
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
		# method: learning method to be used: 'RandomForest', 'XGBoost', 'Linear', or 'Logistic' 
		# seed_start: seed number to be used in the first run, 'seed_start + 1' will be used for the second run, ... 
	
	## 1. Repeat the analysis for 'N_repeat' times
	cv_metric = []
	for i in range(0, N_repeat):
		# split data into 'N_k' folds  
		i_data_split = split_dataset_into_k_folds(data_df, label, N_k, task, seed_no = seed_start + i)		
		# iterate by folds
		i_cv_metric = []
		for ids in i_data_split:
			# obtain split of the fold 
			ids_feat_train, ids_label_train, ids_feat_test, ids_label_test = ids
			# evaluate model on the testing fold 
			_, _, _, _, ids_metric = evaluate_model_performance(ids_feat_train[features].values, ids_feat_test[features].values, ids_label_train.values, ids_label_test.values, task, method)
			i_cv_metric.append(ids_metric)
		# average performance of all folds 
		cv_metric.append(pd.DataFrame(i_cv_metric).mean(axis = 0).to_dict())
	# output as data frame 
	cv_metric_df = pd.DataFrame(cv_metric)

	return cv_metric_df


## This function implements supervised learning model to predict response for new instances
def implement_prediciton_model(model, X_pred, task, pred_prob = 0):
	## 0. Input argument: 
		# model: trained supervised learning model  
		# X_pred: array that contains feature data for prediction
		# task: type of supervised learning task: 'regression' or 'classification' 
		# pred_prob: for classification models, whether to predict class probability (1) or class label (0)
	
	## 1. Choose learning function to implement based on whether the task  
	# regression task 
	if task == 'regression':
		y_pred = model.predict(X_pred)
	# classification task
	if task == 'classification':
		if pred_prob == 1:
			y_pred = model.predict_proba(X_pred)[:,1]
		else:
			y_pred = model.predict(X_pred)

	return y_pred


## This function learns L1 regularized (Lasso) classification/regression from training data, then implements the model on testing data and returns model performance  
def regularize_by_l1(X_train, X_test, y_train, y_test, all_features, N_k, task, N_repeat, seed_no = 0):
	## 0. Input arguments:
		# X_train: array that contains training feature data 
		# X_test: array that contains testing feature data 
		# y_train: array that contains traning response data
		# y_test: array that contains testing response data 
		# all_features: names of all features (column names of X_train)
		# N_k: number of folds to split into 
		# task: type of supervised learning task: 'regression' or 'classification' 
		# N_repeat: number of independent cross-validation runs, each run will generate one performance score
		# seed_no: seed number to be used in the first run, 'seed_start + 1' will be used for the second run, ... 

	## 1. Perform regularized classification/regression based on the specified task 
	# regression
	if task == 'regression':
		# split data into K folds   
		kf = KFold(n_splits = N_k, random_state = seed_no, shuffle = True)
		# find the optimal alpha (regularization factor) using K-fold cross validation on training data 
		cv_regressor = LassoCV(cv = kf, random_state = seed_no)	
		cv_regressor.fit(X_train, y_train)
		best_alpha = cv_regressor.alpha_
		# fit lasso regression using the optimal alpha  
		final_learner = Lasso(alpha = best_alpha)
		final_learner.fit(X_train, y_train) 
		# obtain selected features by fitted lasso regression model (features with coefficients > 0)
		select_features = all_features[(final_learner.coef_ != 0).flatten()]
		N_select = len(select_features)
		# perform K-fold cross validation to obtain the training performance of fitted lasso regression model  
		train_metric = []
		for i in range(0, N_repeat):
			cv_kf = KFold(n_splits = N_k, random_state = i + 1, shuffle = True)
			r2 = cross_val_score(final_learner, X_train, y_train, cv = cv_kf, scoring = 'r2')
			mse = cross_val_score(final_learner, X_train, y_train, cv = cv_kf, scoring = 'neg_mean_squared_error')
			train_metric.append({'r2': np.mean(r2), 'mse': np.mean(mse)})
		train_metric_df = pd.DataFrame(train_metric)
		# implement fitted lasso regression model on the testing set and obtain the testing performance 
		y_pred = final_learner.predict(X_test)
		test_r2 = r2_score(y_test, y_pred)
		test_mse = mean_squared_error(y_test, y_pred)
		test_metric = {'r2': test_r2, 'test_mse': test_mse}	
	
	# classification
	if task == 'classification':
		# straitified split for classification tasks
		kf = StratifiedKFold(n_splits = N_k, random_state = seed_no, shuffle = True)
		# find the optimal C (regularization factor) using K-fold cross validation on training data 
		cv_classifier = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear', cv = kf, random_state = seed_no)
		cv_classifier.fit(X_train, y_train)
		best_c = float(cv_classifier.C_)
		# fit logistic regression using the optimal C
		final_learner = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = best_c, random_state = seed_no)
		final_learner.fit(X_train, y_train)
		# obtain selected features by fitted logistic regression model (features with coefficients > 0)
		select_features = all_features[(final_learner.coef_ != 0).flatten()]
		N_select = len(select_features)
		# perform K-fold cross validation to obtain the training performance of fitted logistic regression model   
		train_metric = []
		for i in range(0, N_repeat):
			cv_kf = StratifiedKFold(n_splits = N_k, random_state = i + 1, shuffle = True)
			auc = cross_val_score(final_learner, X_train, y_train, cv = cv_kf, scoring = 'roc_auc')
			bac = cross_val_score(final_learner, X_train, y_train, cv = cv_kf, scoring = 'balanced_accuracy')
			f1 = cross_val_score(final_learner, X_train, y_train, cv = cv_kf, scoring = 'f1')			
			train_metric.append({'auc': np.mean(auc), 'bac': np.mean(bac), 'f1': np.mean(f1)})
		train_metric_df = pd.DataFrame(train_metric)
		# compare with testing response data, compute metrics
		y_pred_prob = final_learner.predict_proba(X_test)[:,1]
		y_pred = final_learner.predict(X_test)
		test_auc = roc_auc_score(y_test, y_pred_prob) 
		test_bac = balanced_accuracy_score(y_test, y_pred)
		test_f1 = f1_score(y_test, y_pred)
		test_metric = {'auc': test_auc, 'bac': test_bac, 'f1': test_f1}
					
	return final_learner, select_features, train_metric_df, test_metric
