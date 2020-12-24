# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script implements ReBATE methods together with cross-validation to select relevant features, then evaluates model performance on hold-out testing set, eventually implements the model to predict responses of new instances. 


## Module
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/')
import ttox_selection
import ttox_learning


## Main function 
def main(argv):
	## 0. Input arguments 
		# argv 1: input file that contains training feature-response data 
		# argv 2: input file that contains testing feature-response data
		# argv 3: input file that contains prediction feature data 
		# argv 4: prefix of output file name 
		# argv 5: name of label(response) column
		# argv 6: number of folds to split data into
		# argv 7: feature ranking methods to be used: 'SURF', 'SURFstar', 'MultiSURF', or 'MultiSURFstar' 
		# argv 8: whether to implement TURF: 1 or 0 (TURF will remove low-ranking features after each iteration, effective when #features is large)
		# argv 9: percentage of features removed at each iteration (only applied when argv 8 == 1) 
		# argv 10: type of supervised learning task: 'regression' or 'classification'   
		# argv 11: supervised learning methods to be used 'RandomForest' or 'XGBoost' 
		# argv 12: number of performance-decreasing iterations before stopping the model-fitting process (due to overfitting)
		# argv 13: lower bound of percentage to define that a feature is consisently relevant across folds
		# argv 14: number of independent cross-validation runs, each run will generate one performance score
		# argv 15: whether to predict probability of positive class: 1 or 0 (optional, only applied when argv 3 != NA and argv 10 == classification, default: 0)
	output_name = argv[4] + '_fd_' + argv[6] + '_fr_' + argv[7] + '_tf_' + argv[8] + '_pc_' + argv[9] + '_md_' + argv[11] + '_tl_' + argv[12] + '_cs_' + argv[13] + '_nr_' + argv[14]
		
	## 1. Read in input training and testing files 
	# read in training data
	train_data_df = pd.read_csv(argv[1], sep = '\t', header = 0, index_col = 0)
	# read in testing data 
	test_data_df = pd.read_csv(argv[2], sep = '\t', header = 0, index_col = 0)

	## 2. Split training feature-response dataset into K folds 
	label_col_name = argv[5]
	train_data_split_list = ttox_learning.split_dataset_into_k_folds(train_data_df, label_col_name, int(argv[6]), argv[10], seed_no = 0)
	
	## 3. Compute feature importance scores on each fold of training data using ReBATE methods  
	feature_importance_df = ttox_selection.rank_features_by_rebate_methods(train_data_split_list, argv[7], int(argv[8]), remove_percent = float(argv[9]))
	feature_importance_df.to_csv(output_name + '_importance.tsv', sep = '\t', float_format = '%.5f')
	
	## 4. Identifies relevant features on each fold of training data based on feature importance scores   
	feature_select_id_df = ttox_selection.identify_relevant_features(train_data_split_list, argv[10], argv[11], feature_importance_df, int(argv[12]))

	## 5. Selects features that are consistently relevant in cross-validation process
	feature_consistency_pct, select_features = ttox_selection.select_consistent_features(feature_select_id_df, float(argv[13]))
	feature_select_df = pd.concat([feature_select_id_df, feature_consistency_pct], axis = 1)
	feature_select_df.to_csv(output_name + '_select.tsv', sep = '\t', float_format = '%.5f')

	## 6. Compute model performance on training data by cross-validation (for hyperparameter tuning)
	metric_select_train = [np.nan]
	if len(select_features) > 0:
		metric_select_train = ttox_learning.evaluate_model_performance_by_cv(train_data_df, select_features, label_col_name, int(argv[14]), int(argv[6]), argv[10], argv[11], seed_start = 1)
	
	## 7. Builds supervised learning model from all training data, then evaluates model performance on hold-out testing data tests the model on testing data
	# use all features to build the model 
	N_all_features, N_train_instances, N_test_instances, model_all, metric_all_test = ttox_learning.evaluate_model_performance(train_data_df.drop(label_col_name, axis = 1).values, test_data_df.drop(label_col_name, axis = 1).values, train_data_df[label_col_name].values, test_data_df[label_col_name].values, argv[10], argv[11], seed_number = 0)
	# use selected features to build the model 
	N_select_features, metric_select_test = 0, np.nan
	if len(select_features) > 0:
		N_select_features, N_train_instances, N_test_instances, model_select, metric_select_test = ttox_learning.evaluate_model_performance(train_data_df[select_features].values, test_data_df[select_features].values, train_data_df[label_col_name].values, test_data_df[label_col_name].values, argv[10], argv[11], seed_number = 0)	
	# generate performance summary file  
	perf_summary = ttox_selection.generate_performance_summary(N_train_instances, N_test_instances, N_all_features, metric_all_test, select_features, N_select_features, metric_select_train, metric_select_test)
	perf_file = open(output_name + '_performance.txt', 'w')
	for ps in perf_summary:
		perf_file.write('%s\n' % ps)
	perf_file.close()

	## 8. Implements learned model to predict response for new instances
	if argv[3] != 'NA':
		# whether to predict probability of positive class
		prob_indi = 0
		if len(argv) > 15: 
			prob_indi = int(argv[15])
		pred_data_df = pd.read_csv(argv[3], sep = '\t', header = 0, index_col = 0)
		# use all features to predict 
		pred_label_all = ttox_learning.implement_prediciton_model(model_all, pred_data_df.values, argv[10], pred_prob = prob_indi)
		# use selected featuers to predict 
		pred_label_select = np.empty(pred_data_df.shape[0])
		if len(select_features) > 0:
			pred_label_select = ttox_learning.implement_prediciton_model(model_select, pred_data_df[select_features].values, argv[10], pred_prob = prob_indi)
		# Aggregate predictions into one data frame (2 columns, 1: predictions from model using selected features, 2: predictions from model using all features) 
		pred_df = pd.DataFrame({'select_features_pred': pred_label_select,'all_features_pred': pred_label_all}, index = pred_data_df.index)
		pred_df.to_csv(output_name + '_prediction.tsv', sep = '\t', float_format = '%.5f')	

	return 1


## Call main function
main(sys.argv) 
