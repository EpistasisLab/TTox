# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script analyzes and visualizes the results of feature selection pipeline.  


## Module
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/')
import ttox_selection 
import ttox_plot


## Main function
def main(argv):
	## 0. Input arguments: 
		# argv 1: input file that contains performance file names 
		# argv 2: type of supervised learning task: 'regression' or 'classification' 
		# argv 3: name of metric that is used for model selection 
		# argv 4: type of analysis task: 'tuning' for hyperparameter tuning results, 'implementation' for pipeline implementation results 
		# argv 5: output file of analysis results
		# argv 6: output file of visualization results 
		# argv 7: input file that contains performance file names of L1-regularized models (only needed when argv 4 == 'implementation') 
		# argv 8: input file that contains number of selected features of models (only needed when argv 4 == 'implementation')
		# argv 9: threshold of tesing performance for model to be considered (only needed when argv 4 == 'implementation') 
		# argv 10: prefix of prediction files (only needed when argv 4 == 'implementation') 
		# argv 11: prefix of performance files (only needed when argv 4 == 'implementation') 
			
	## 1. Read in input performance files 
	model_file_df = pd.read_csv(argv[1], sep = '\t', header = 0, index_col = 0)

	## 2. Perform analysis according to the specified task  
	# analyze performance results of hyperparamter tuning 
	if argv[4] == 'tuning':
		# read in training performance of models under different hyperparamter settings 
		select_train_file = model_file_df.loc['select_train', argv[3]]
		select_feat_train_df = pd.read_csv(select_train_file, sep = '\t', header = 0, index_col = 0)
		# find the optimal hyperparameter setting by maximum median of all models 
		select_feat_train_df1, optimal_hp_col, optimal_hp_list = ttox_selection.find_optimal_hyperparameter_setting(select_feat_train_df, 'median', argv[2])
		# output the optimal hyperparameter setting
		optimal_file = open(argv[5] + '_optimal_hyperparameters.txt', 'w')
		for ohl in optimal_hp_list:
			optimal_file.write('%s\n' % ohl)
		optimal_file.close()
	
		# visualize performance comparison under different hyperparameter settings 
		if optimal_hp_col.split('_')[0] == 'fd':
			visualize_hp = ttox_plot.visualize_hyperparameter_comparison(select_feat_train_df1, argv[2], argv[6])

	# analyze performance results of implementation  
	if argv[4] == 'implementation':
		# read in testing performance of models under optimal hyperparamter settings
		select_test_file = model_file_df.loc['select_test', argv[3]]
		select_feat_test_df = pd.read_csv(select_test_file, sep = '\t', header = 0, index_col = 0)
		# select the model with the best performance for each target 
		optimal_col = select_feat_test_df.columns[0]
		select_feat_test_df1 = ttox_selection.select_target_measurement_by_performance(select_feat_test_df, optimal_col)
		select_feat_test_df1.to_csv(argv[5] + '_testing_performance_summary_' + argv[3] + '_select.tsv', sep = '\t', float_format = '%.5f')
		select_models = select_feat_test_df1.index
		# visualize comparison among three model sets: 1) model built on selected features, ii) model built on all features, and iii) model built with L1 regularization  
		visualize_compare = ttox_plot.visualize_testing_performance_comparison(model_file_df, argv[2], select_models, argv[6], argv[7])
		
		# read in testing performance of models built on all features 
		all_test_file = model_file_df.loc['all_test', argv[3]]
		all_feat_test_df = pd.read_csv(all_test_file, sep = '\t', header = 0, index_col = 0)
		# read in data frame that contains number of selected features of models
		select_feat_number_df = pd.read_csv(argv[8], sep = '\t', header = 0, index_col = 0)
		# computes basic statistics of two model sets: 1) model built on selected features and ii) model built on all features 
		selection_stat = ttox_selection.compute_feature_selection_statistic(all_feat_test_df.loc[select_models,], select_feat_test_df1, select_feat_number_df.loc[select_models,], optimal_col, float(argv[9]))
		stat_file = open(argv[5] + '_mc_' + argv[9] + '_feature_selection_statistics.txt', 'w')
		for ss in selection_stat:
			stat_file.write('%s\n' % ss)
		stat_file.close()
		
		# collect predictions from models under the optimal hyperparameter setting and models without feature selection  
		select_pred_df, all_pred_df = ttox_selection.collect_model_prediction(select_feat_test_df1, optimal_col, float(argv[9]), argv[10])
		select_pred_df.to_csv(argv[5] + '_mc_' + argv[9] + '_offsides_compounds_binding_affinity_prediction_select_features.tsv', sep = '\t', float_format = '%.5f')
		all_pred_df.to_csv(argv[5] + '_mc_' + argv[9] + '_offsides_compounds_binding_affinity_prediction_all_features.tsv', sep = '\t', float_format = '%.5f')
		
		# connect targets to relevant structure features from models with best performance 
		target_structure_df, tm_structure_df = ttox_selection.connect_structure_target(select_feat_test_df1, optimal_col, float(argv[9]), argv[11])
		target_structure_df.to_csv(argv[5] + '_mc_' + argv[9] + '_target_structure.tsv', sep = '\t', index = False, header = False)		
		tm_structure_df.to_csv(argv[5] + '_mc_' + argv[9] + '_target_measurement_structure.tsv', sep = '\t', index = False)  

	return 1
 

## Call main function
main(sys.argv)				 		 
