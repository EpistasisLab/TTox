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
	
	return 1
 

## Call main function
main(sys.argv)				 		 
