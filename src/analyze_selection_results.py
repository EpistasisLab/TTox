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
		# argv 1: input file that contains performance of models using all features (row: dataset)
		# argv 2: input file that contains training performance of models under different hyperparamter settings (row: dataset, column: setting)
		# argv 3: input file that contains testing performance of models under different hyperparamter settings (row: dataset, column: setting) 
		# argv 4: type of analysis task: 'tuning' for hyperparameter tuning results, 'implementation' for pipeline implementation results 
		# argv 5: output file of analysis results
		# argv 6: output file of visualization results 
		# argv 7: input file that contains number of selected features of models (only needed when argv 5 == 'implementation')
		# argv 8: threshold of tesing performance for model to be considered (only needed when argv 5 == 'implementation') 
		# argv 9: prefix of prediction files (only needed when argv 5 == 'implementation') 
		# argv 10: prefix of performance files (only needed when argv 5 == 'implementation') 
		
	## 1. Read in input performance files 
	all_feat_test_df = pd.read_csv(argv[1], sep = '\t', header = 0, index_col = 0) 
	select_feat_train_df = pd.read_csv(argv[2], sep = '\t', header = 0, index_col = 0)
	select_feat_test_df = pd.read_csv(argv[3], sep = '\t', header = 0, index_col = 0)

	## 2. Perform analysis according to the specified task  
	# analyze performance results of hyperparamter tuning 
	if argv[4] == 'tuning':
		# find the optimal hyperparameter setting by maximum median of all models 
		select_feat_train_df1, optimal_hp_col, optimal_hp_list = ttox_selection.find_optimal_hyperparameter_setting(select_feat_train_df, 'median')
		# output the optimal hyperparameter setting
		optimal_file = open(argv[5] + '_optimal_hyperparameters.txt', 'w')
		for ohl in optimal_hp_list:
			optimal_file.write('%s\n' % ohl)
		optimal_file.close()
		# visualize performance comparison under different hyperparameter settings 
		visualize_hp = ttox_plot.visualize_hyperparameter_comparison(select_feat_train_df1, argv[6])
		# visualizes testing performance comparison between model using all features and optimal model using selected features 
		all_feat_test_perf = all_feat_test_df.loc[select_feat_train_df1.index, 'all_features_testing'].values
		select_feat_test_perf = select_feat_test_df.loc[select_feat_train_df1.index, optimal_hp_col].values
		visualize_testing = ttox_plot.visualize_testing_performance_comparison(all_feat_test_perf, select_feat_test_perf, argv[6])

	# analyze performance results of implementation  
	if argv[4] == 'implementation':
		# read in data frame that contains number of selected features of models
		select_feat_number_df = pd.read_csv(argv[7], sep = '\t', header = 0, index_col = 0)	
		# select the model with the best performance for each target
		optimal_col = select_feat_test_df.columns[0] 
		select_feat_test_df1 = ttox_selection.select_target_measurement_by_performance(select_feat_test_df, optimal_col)		
		# visualize testing performance comparison between model using all features and optimal model built upon selected features 
		all_feat_test_perf = all_feat_test_df.loc[select_feat_test_df1.index, 'all_features_testing'].values
		select_feat_test_perf = select_feat_test_df1[optimal_col].values
		visualize_testing = ttox_plot.visualize_testing_performance_comparison(all_feat_test_perf, select_feat_test_perf, argv[6])
		# compute basic statistics of models 
		selection_stat = ttox_selection.compute_feature_selection_statistic(all_feat_test_df.loc[select_feat_test_df1.index,], select_feat_test_df1, select_feat_number_df.loc[select_feat_test_df1.index,], optimal_col, float(argv[8]))
		stat_file = open(argv[5] + '_auc_' + argv[8] + '_feature_selection_statistics.txt', 'w')
		for ss in selection_stat:
			stat_file.write('%s\n' % ss)
		stat_file.close()
		# collect predictions from models under the optimal hyperparameter setting and models without feature selection  
		select_pred_df, all_pred_df = ttox_selection.collect_model_prediction(select_feat_test_df1, optimal_col, float(argv[8]), argv[9])
		select_pred_df.to_csv(argv[5] + '_auc_' + argv[8] + '_offsides_compounds_binding_affinity_prediction_select_features.tsv', sep = '\t', float_format = '%.5f')
		all_pred_df.to_csv(argv[5] + '_auc_' + argv[8] + '_offsides_compounds_binding_affinity_prediction_all_features.tsv', sep = '\t', float_format = '%.5f')
		# connect targets to relevant structure features from models with best performance 
		target_structure_df, tm_structure_df = ttox_selection.connect_structure_target(select_feat_test_df1, optimal_col, float(argv[8]), argv[10])
		target_structure_df.to_csv(argv[5] + '_auc_' + argv[8] + '_target_structure.tsv', sep = '\t', index = False, header = False)		
		tm_structure_df.to_csv(argv[5] + '_auc_' + argv[8] + '_target_measurement_structure.tsv', sep = '\t', index = False)  

	return 1
 

## Call main function
main(sys.argv)				 		 
