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
from scipy import stats
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
	# define metric used for feature selection according to the specified task  
	if ml_task == 'regression':
		select_metric = 'r2'
	if ml_task == 'classification':
		select_metric = 'auc'
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
			# intermediate output 
			if verbose == True:
				print('Fitting model for feature ' + str(j) + '\n')
			# extract the top j features from training/testing data
			j_feat = sorted_feat[0:j]
			j_feat_train = feat_train[j_feat]
			j_feat_test = feat_test[j_feat]
			# perform regression/classification task using the top j features
			_, _, _, _, j_metric_dict = ttox_learning.evaluate_model_performance(j_feat_train.values, j_feat_test.values, label_train.values, label_test.values, ml_task, ml_method)
			j_metric = j_metric_dict[select_metric]
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


## This function converts model metrics to a string
def convert_metric_to_string(metric_dict, round_digit = 5):
	## 0. Input arguments: 
		# metric_dict: dictionary that contains metrics of model performance 
		# round_digit: number of decimal places to round to (default: 5) 
	
	## 1. Join metric names and values to build output strings 
	# iterate by item in metric_dict 
	metric_str = []
	for k,v in metric_dict.items():
		# round metric values  
		v = np.round(v, round_digit)
		# convert metric values to strings  
		if type(v) is np.ndarray:
			vv_str = [str(vv) for vv in v]	
			v_str = ','.join(vv_str)
		else:
			v_str = str(v)
		# join the metric name 
		metric_str.append(k + ':' + v_str)
	# join all item strings together 
	output_str = ';'.join(metric_str)
	
	return output_str
	

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

	## 1. Add model statistics to a list  
	perf_list = []
	perf_list.append('Number of training instances: ' + str(N_train_instances))
	perf_list.append('Number of testing instances: ' + str(N_test_instances))
	perf_list.append('Number of all features: ' + str(N_all_features))
	perf_list.append('Testing performance of model using all features: ' + convert_metric_to_string(metric_all_test))
	perf_list.append('Relevant features: ' + ','.join(select_features))
	perf_list.append('Number of relevant features: '+ str(N_select_features))
	perf_list.append('Training performance of model using relevant features: ' + convert_metric_to_string(metric_select_train.to_dict(orient = 'list')))
	perf_list.append('Testing performance of model using relevant features: '+ convert_metric_to_string(metric_select_test))

	return perf_list


## This function finds the optimal hyperparamter setting based on model performance
def find_optimal_hyperparameter_setting(perf_df, method, task):
	## 0. Input arguments:
		# perf_df: data frame containing model performance under different hyperparamter settings (row: model, column: hyperparameter)   
		# method: metric for defining optimal hyperparamter setting: 'median' or 'mean'
		# task: type of supervised learning task: 'regression' or 'classification'
	
	## 1. Remove models with no selected features under some hyperparameter settings (rows with NA's values)
	perf_df = perf_df[perf_df.isna().sum(axis = 1) == 0]

	## 2. Find optimal hyperparamter setting according to specific method   
	# based on maximum median value
	if method == 'median':
		optimal_hs = perf_df.median(axis = 0).sort_values(ascending = False).index[0]	
	# based on maximum average value 
	if method == 'mean':
		optimal_hs = perf_df.mean(axis = 0).sort_values(ascending = False).index[0]

	## 3. Build output list that contains detailed optimal hyperparameter setting 
	optimal_list = []
	optimal_hs_s = optimal_hs.split('_')
	# optimal hyperparameter setting for structure-target datasets 
	if optimal_hs_s[0] == 'fd':
		optimal_list.append('Number of folds: 10')
		optimal_list.append('Feature ranking method: ' + optimal_hs_s[1])
		optimal_list.append('Implement TURF: ' + optimal_hs_s[3])
		optimal_list.append('Remove percentile: ' + optimal_hs_s[5])
		optimal_list.append('Learning task: ' + task)
		optimal_list.append('Classification/regression method: ' + optimal_hs_s[7])
		optimal_list.append('Tolerance iterations: ' + optimal_hs_s[9])
		optimal_list.append('Threshold of consistency ratio: ' + optimal_hs_s[11])
		optimal_list.append('Number of repeats: 20')
	# optimal hyperparameter setting for target-adverse event datasets  
	if optimal_hs_s[0] == 'mc': 
		optimal_list.append('Target binding profile AUC threshold: ' + optimal_hs_s[1])
		optimal_list.append('Number of folds: 10')
		optimal_list.append('Feature ranking method: ' + optimal_hs_s[5])
		optimal_list.append('Implement TURF: ' + optimal_hs_s[7])
		optimal_list.append('Remove percentile: ' + optimal_hs_s[9])
		optimal_list.append('Learning task: ' + task)
		optimal_list.append('Classification/regression method: ' + optimal_hs_s[11])
		optimal_list.append('Tolerance iterations: 50')
		optimal_list.append('Threshold of consistency ratio: 0.5')
		optimal_list.append('Number of repeats: 20')

	return perf_df, optimal_hs, optimal_list


## This function selects measurements of targets by model performance. Each target will be assigned a single measurement-model with the best performance.  
def select_target_measurement_by_performance(perf_df, optimal_column):
	## 0. Input arguments:
		# perf_df: data frame containing testing performance under different hyperparameter settings (row: model, column: hyperparameter) 
		# optimal_column: column of perf_df that contains the results of optimal hyparameter setting  
	
	## 1. Remove models with no selected features under some hyperparameter settings (rows with NA's values)
	perf_df = perf_df[perf_df.isna().sum(axis = 1) == 0]	

	## 2. Obtain the intended target of each model 
	# iterate by model
	targets = []
	for pdi in perf_df.index:
		targets.append(pdi.split('_')[0])
	# obtain the unique target set
	uni_targets = np.unique(targets)
	
	## 3. Select the measurement with the best performance for each target
	# iterate by target
	select_row_id = [] 
	for ut in uni_targets:
		# find the row ID of all measurements with the intended target 
		ut_id = [index for index, value in enumerate(targets) if value == ut] 		
		# identify the measurement with the best performance 
		ut_select = perf_df[optimal_column].iloc[ut_id,].idxmax(axis = 0)
		select_row_id.append(ut_select)
	# select the rows containing models of best performance  
	select_perf_df = perf_df.loc[select_row_id,:]

	return select_perf_df

	
## This function computes mean of an array, and confidence interval of mean by bootstrapping.
def compute_mean_and_ci_by_bootsrap(vec, confidence_interval = 0.95, bootstrap_times = 1000):
	## 0. Input arguments: 
		# vec: input array 
		# confidence_interval: confidence interval to be computed (number between 0 and 1)
		# bootstrap_times: repeated sampling times for bootstrap
	
	## 1. Compute mean of array
	vec_mean = np.mean(vec)
        
	## 2. Compute confidence interval of mean by bootstrapping
	vec_len = len(vec) 
	# Repeat boostrap process
	sample_means = []
	for sample in range(0, bootstrap_times):
		# Sampling with replacement from the input array
		sample_values = np.random.choice(vec, size = vec_len, replace = True)
		# compute sample mean
		sample_mean = np.mean(sample_values)
		sample_means.append(sample_mean)
	# sort means of bootstrap samples 
	sample_means = np.sort(sample_means)
	# obtain upper and lower index of confidence interval 
	lower_id = int((0.5 - confidence_interval/2) * bootstrap_times) - 1
	upper_id = int((0.5 + confidence_interval/2) * bootstrap_times) - 1

	return vec_mean, sample_means[lower_id], sample_means[upper_id]


## This function tests whether the mean of one array is significantly greater than the mean of another array 
def compare_sample_means_by_one_sided_test(vec1, vec2, parametric, related): 
	## 0. Input arguments: 
		# vec1: array 1 of 2 to be compared 
		# vec2: array 2 of 2 to be compared (alternative hypothesis: mean of vec1 > mean of vec2)
		# parametric: whether to use parametric test (True) or non-parameteric test (False)
		# related: whether two arrays are related (True) or not (False)

	## 1. Perform two-sided statistical test according to the specified parameters 
	# two-sided parametric test 
	if parametric == True:
		# t test with related samples 
		if related == True:  
			two_sided_p = stats.ttest_rel(vec1, vec2)[1]
		# t test with independent samples
		else:
			two_sided_p = stats.ttest_ind(vec1, vec2)[1]
	# two-sided nonparametric test 	
	else:
		# Wilcoxon signed-rank test 
		if related == True: 
			two_sided_p = stats.wilcoxon(vec1, vec2)[1]	
		# Mann–Whitney U test 
		else: 
			two_sided_p = stats.mannwhitneyu(vec1, vec2)[1]
	
	## 2. Covert two-sided test p-value to one-sided test p-value by comparing the average  
	# compute mean of each array  
	vec1_mean = np.mean(vec1)
	vec2_mean = np.mean(vec2)
	# convert p-value 
	if vec1_mean > vec2_mean:
		one_sided_p = two_sided_p/2
	else:
		one_sided_p = 1 - two_sided_p/2
	
	return	one_sided_p 


## This function computes basic statistics of feature selection results
def compute_feature_selection_statistic(all_perf_df, select_perf_df, select_number_df, optimal_column, perf_threshold):
	## 0. Input arguments: 
		# all_perf_df: data frame containing testing performance of models built upon all features  
		# select_perf_df: data frame containing testing performance under different hyperparameter settings (row: model, column: hyperparameter) 
		# select_number_df: data frame containing number of selected features under different hyperparameter 
		# optimal_column: column of select_perf_df and select_number_df that contains the results of optimal hyparameter setting 
		# perf_threshold: threshold of tesing performance for model to be considered 

	## 1. Select rows containing models with performance better than threshold
	select_perf_df = select_perf_df[select_perf_df[optimal_column] >= perf_threshold]
	all_perf_df = all_perf_df.loc[select_perf_df.index, ]
	select_number_df = select_number_df.loc[select_perf_df.index, ]

	## 2. Obtain statistics of feature selection 
	# number of models with testing performance better than the specified threshold 
	N_models = select_perf_df.shape[0]
	# number of all features 
	N_all = all_perf_df['N_all_features'].values[0]
	# average testing performance and 95% CI of models built upon all features
	all_perf_mean, all_perf_ci_lower, all_perf_ci_upper = compute_mean_and_ci_by_bootsrap(all_perf_df[optimal_column].values)
	# average number and 95% CI of models built upon selected features
	N_select_mean, N_select_ci_lower, N_select_ci_upper = compute_mean_and_ci_by_bootsrap(select_number_df[optimal_column].values)
	# average testing performance and 95% CI of models built upon selected features
	select_perf_mean, select_perf_ci_lower, select_perf_ci_upper = compute_mean_and_ci_by_bootsrap(select_perf_df[optimal_column].values)
	# p-value of one-sided Wilcoxon signed-rank test to examine whether there is a difference bewteen testing performance of models built upon all features and selected features
	one_sided_p = compare_sample_means_by_one_sided_test(select_perf_df[optimal_column].values, all_perf_df[optimal_column].values, False, True)

	## 3. Build output list of statistics
	fs_stat = []
	fs_stat.append('Number of target-models: '+ str(N_models))
	fs_stat.append('Number of all features: ' + str(N_all))
	fs_stat.append('Average (95% CI) performance of models built upon all features: ' + str(round(all_perf_mean, 3)) + '(' + str(round(all_perf_ci_lower, 3)) + '-' + str(round(all_perf_ci_upper, 3)) + ')')
	fs_stat.append('Average number (95% CI) of selected features: ' + str(round(N_select_mean, 0)) + '(' + str(round(N_select_ci_lower, 0)) + '-' + str(round(N_select_ci_upper, 0)) + ')')
	fs_stat.append('Average (95% CI) performance of models built upon selected features: ' + str(round(select_perf_mean, 3)) + '(' + str(round(select_perf_ci_lower, 3)) + '-' + str(round(select_perf_ci_upper, 3)) + ')')
	fs_stat.append('One-sided paired Wilcoxon signed-rank test p-value: ' + str(one_sided_p))

	return fs_stat


## This function collects predictions from models under the optimal hyperparameter setting and models with feature selection 
def collect_model_prediction(perf_df, optimal_column, perf_threshold, pred_folder):
	## 0. Input arguments: 
		# perf_df: data frame containing testing performance under different hyperparameter settings (row: model, column: hyperparameter) 
		# optimal_column: column of perf_df that contains the results of optimal hyparameter setting  
		# perf_threshold: threshold of tesing performance for model to be considered 
		# pred_folder: folder of prediction files

	## 1. Select rows containing models with performance better than threshold
	perf_df = perf_df[perf_df[optimal_column] >= perf_threshold]
	
	## 2. Collect the prediction of models under optimal hyperparameter setting
	# iterate by model 
	hp_pred_list = []
	all_pred_list = []
	targets = []
	for pdi in perf_df.index:
		# obtain the name of prediction file  
		pdi_target = pdi.split('_')[0]
		pdi_measure = pdi.split('_')[1]
		pdi_pred_file = pred_folder + '_' + pdi_measure + '_0.25_binary_' + pdi_target + '_whole_data.tsv_fd_10_' + optimal_column + '_nr_20_prediction.tsv'
		targets.append(pdi_target)
		# read in the prediction of models and select the column of interest 
		pdi_pred_df = pd.read_csv(pdi_pred_file, sep = '\t', header = 0, index_col = 0)				
		# obtain prediction from model with feature selection 
		hp_pred_list.append(pdi_pred_df['select_features_pred']) 
		# obtain prediction from model without feature selection 
		all_pred_list.append(pdi_pred_df['all_features_pred'])
	# aggregate prediction results of all models 
	hp_pred_df = pd.concat(hp_pred_list, axis = 1)
	hp_pred_df.columns = targets
	all_pred_df = pd.concat(all_pred_list, axis = 1)
	all_pred_df.columns = targets	

	return hp_pred_df, all_pred_df


## This function connects targets to relevant structure features from models with best performance 
def connect_structure_target(perf_df, optimal_column, perf_threshold, perf_folder):
	## 0. Input arguments: 
		# perf_df: data frame containing testing performance under different hyperparameter settings (row: model, column: hyperparameter) 
		# optimal_column: column of perf_df that contains the results of optimal hyparameter setting  
		# perf_threshold: threshold of tesing performance for model to be considered  
		# perf_folder: folder of performance files

	## 1. Select rows containing models with performance better than threshold 
	perf_df = perf_df[perf_df[optimal_column] >= perf_threshold]
			
	## 2. Collect the prediction of models under optimal hyperparameter setting
	# iterate by model 
	selected_features = []
	targets = []
	for pdi in perf_df.index:
		# obtain the name of performance file 
		pdi_target = pdi.split('_')[0]
		pdi_measure = pdi.split('_')[1] 
		pdi_perf_file = perf_folder + '_' + pdi_measure + '_0.25_binary_' + pdi_target + '_whole_data.tsv_fd_10_' + optimal_column + '_nr_20_performance.txt'
		targets.append(pdi_target)
		# read in the performance file  
		pdi_perf_file_read = open(pdi_perf_file, 'r')
		for line in pdi_perf_file_read:
			line = line.strip()
			# find the line that contains relevant features
			if line.startswith('Relevant features: '):
				# obtain relevant features 
				features = line.split('Relevant features: ')[1]
				selected_features.append(features)
		pdi_perf_file_read.close()
	# output target-feature reletionships in data frame format  
	target_structure_df = pd.DataFrame({'target': targets, 'structure_features': selected_features})
	tm_structure_df = pd.DataFrame({'target_measurement': perf_df.index, 'structure_features': selected_features})
	
	return	target_structure_df, tm_structure_df
