# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains functions for selecting relevant features by cross-validation.


## Module
import sys
import numpy as np
import pandas as pd
from skrebate import SURF
from skrebate import SURFstar
from skrebate import MultiSURF
from skrebate import MultiSURFstar
from skrebate import TuRF
sys.path.insert(0, 'src/')
import ttox_learning


## This function compute feature importance scores on each fold of training data using ReBATE methods  
def rank_features_by_rebate_methods(data_split_list, fs_method, iterate, remove_percent = 0.1, verbose = False):
        ## 0. Input arguments: 
                # data_split_list: data frame that contains the learning data
		# fs_method: feature ranking methods to be used: 'SURF', 'SURFstar', 'MultiSURF', or 'MultiSURFstar' 
		# iterate: whether to implement TURF: True or False (TURF will remove low-ranking features after each iteration, effective when #features is large)
		# remove_percent: percentage of features removed at each iteration (only applied when iterate = True)
		# verbose: whether to show progress by each fold: True of False
	
	## 1. Define function for feature ranking method 
	# SURF
	if fs_method == 'SURF':
		# Implement TURF extension when 'iterate == True' 
		if iterate == True: 
			fs = TuRF(core_algorithm = 'SURF', pct = remove_percent)
		else:
			fs = SURF()
	# SURFstar
	if fs_method == 'SURFstar':
		if iterate == True:
			fs = TuRF(core_algorithm = 'SURFstar', pct = remove_percent)
		else:
			fs = SURFstar()
	# MultiSURF
	if fs_method == 'MultiSURF':
		if iterate == True:
			fs = TuRF(core_algorithm = 'MultiSURF', pct = remove_percent)
		else:
			fs = MultiSURF() 
	# MultiSURFstar
	if fs_method == 'MultiSURFstar':
		if iterate == True:
			fs = TuRF(core_algorithm = 'MultiSURFstar', pct = remove_percent)
		else:
			fs = MultiSURFstar()

	## 2. Perform feature ranking on each fold of training data 
	# iterate by folds
	feat_impt_dict = {}
	for i in range(0, len(data_split_list)):
		# intermediate output 
		if verbose == True:
			print('Computing feature importance scores using data from fold ' + str(i) + '\n')
                # obtain training feature matrix and response vector 
		feat_train, label_train, _, _ = data_split_list[i]
		# fit feature ranking model using the specified method
		if iterate == True:
			fs.fit(feat_train.values, label_train.values, list(feat_train))
		else: 
			fs.fit(feat_train.values, label_train.values)
		# output feature importance scores in a data frame 
		fold_name = 'Fold_' + str(i)
		feat_impt_dict[fold_name] = fs.feature_importances_
	# aggregate results from muliple folds into one data frame 
	feat_impt_df = pd.DataFrame(feat_impt_dict)
	feat_impt_df.index = feat_train.columns
	
	return feat_impt_df


## This function identifies relevant features on each fold of training data based on feature importance scores   
def identify_relevant_features(data_split_list, ml_task, ml_method, feat_impt_df, tolerance, verbose = False):
	## 0. Input arguments: 
		# data_split_list: data frame that contains the learning data
		# ml_task: type of supervised learning task: 'regression' or 'classification'  
		# ml_method: supervised learning methods to be used 'RandomForest' or 'XGBoost'
		# feat_impt_df: data frame that contains feature importance scores (column: fold of training data, row: feature) 
		# tolerance: number of performance-decreasing iterations before stopping the model-fitting process (due to overfitting)
		# verbose: whether to show progress by each fold: True of False
	
	## 1.Identify relevant features of each fold 
	# iterate by fold
	select_ids = []
	for i in range(0, len(data_split_list)):
		# intermediate output 
		if verbose == True:
			print('Identifying relevant features using data from fold ' + str(i) + '\n')
		# obtain training feature matrix and response vector of the fold 
		feat_train, label_train, feat_test, label_test = data_split_list[i]	
		# obtain sorted feature list of the fold 
		fold_col = feat_impt_df.columns[i]
		sorted_feat = feat_impt_df.sort_values([fold_col], ascending = False).index
		# initialize variable to keep track of feature iteration 
		j = 1
		# initialize variable to keep track of performance-decreasing iterations   
		j_tol = 0 
		# initialize variables to keep track of best performance iterations/models 
		j_best = -1
		metric_best = -float('inf')
		# iterate by feature ranking before finding the best-performing featue set
		while (j_tol < tolerance) & (j < len(sorted_feat) + 1): 
			# extract the top j features from training/testing data
			j_feat = sorted_feat[0:j]
			j_feat_train = feat_train[j_feat]
			j_feat_test = feat_test[j_feat]
			# perform regression/classification task using the top j features
			_, _, _, _, j_metric = ttox_learning.evaluate_model_performance(j_feat_train.values, j_feat_test.values, label_train.values, label_test.values, ml_task, ml_method)
			# compare performance of current iteration with best performance from previous iterations
			if j_metric > metric_best:
				# find a better-performing feature set 
				metric_best, j_best = j_metric, j
				j_tol = 0
			else: 
				j_tol = j_tol + 1
			# go to next iteration
			j = j + 1 
		# build binary vector indicating whether each feature is in the best-performing set (0: irrelevant, 1: relevant)
		select_id = pd.DataFrame(np.zeros(len(sorted_feat), dtype = int), columns = ['Fold_' + str(i)], index = feat_impt_df.index) 
		best_feat = sorted_feat[0:j_best]
		select_id.loc[best_feat, ] = 1
		select_ids.append(select_id)
	
	## 2. Aggregate feature indicators from all folds
	select_ids_df = pd.concat(select_ids, axis = 1)			
	
	return select_ids_df	


## This function selects features that are consistently relevant in cross-validation process
def select_consistent_features(select_ids_df, pct_cut):
	## 0. Input arguments: 
		# select_ids_df: data frame that contains binary vector indicating whether each feature is in the best-performing set (column: fold of training data, row: feature)
		# pct_cut: lower bound of percentage to define that a feature is consisently relevant across folds  
	
	## 1. Compute the percentage of folds in which a feature is relevant
	select_ids_mean = select_ids_df.mean(axis = 1)
	select_ids_mean_df = pd.DataFrame(select_ids_mean, columns = ['consistency_score'])
	
	## 2. Select consistent features by threshold of percentage 
	select_features = select_ids_df[select_ids_mean >= pct_cut].index

	return select_ids_mean_df, select_features


## This function generates a list that describes summary statistics of a supervised learning model 
def generate_performance_summary(N_train_instances, N_test_instances, N_all_features, metric_all_test, select_features, N_select_features, metric_select_train, metric_select_test):
	## 0. Input arguments: 
		# N_train_instances: number of training instances
		# N_test_instances: number of testing instances
		# N_all_features: number of all features
		# metric_all_test: testing performance of model using all features
		# select_features: names of relevant features 
		# N_select_features: number of relevant features 
		# metric_select_train: training performance of model using relevant features 
		# metric_select_test: testing performance of model using relevant features 

	## 1. Convert training performance list to string 
	metric_select_train = np.round(metric_select_train, 5)
	metric_select_train_str = [str(mst) for mst in metric_select_train]
	
	## 2. Add model statistics to a list  
	perf_list = []
	perf_list.append('Number of training instances: ' + str(N_train_instances))
	perf_list.append('Number of testing instances: ' + str(N_test_instances))
	perf_list.append('Number of all features: ' + str(N_all_features))
	perf_list.append('Testing performance of model using all features: ' + str(round(metric_all_test, 5)))
	perf_list.append('Relevant features: ' + ','.join(select_features))
	perf_list.append('Number of relevant features: '+ str(N_select_features))
	perf_list.append('Training performance of model using relevant features: ' + ','.join(metric_select_train_str))
	perf_list.append('Testing performance of model using relevant features: '+ str(round(metric_select_test, 5)))

	return perf_list
