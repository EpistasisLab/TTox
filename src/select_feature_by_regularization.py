# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script builds L1 regularized classification/regression models to select relevant features, then evaluates model performance on hold-out testing set, eventually implements the model to predict responses of new instances. 


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
		# argv 7: type of supervised learning task: 'regression' or 'classification'   
		# argv 8: number of independent cross-validation runs, each run will generate one performance score
	output_name = argv[4] + '_fd_' + argv[6] + '_nr_' + argv[8]
		
	## 1. Read in input training and testing files 
	# read in training data
	train_data_df = pd.read_csv(argv[1], sep = '\t', header = 0, index_col = 0)
	# read in testing data 
	test_data_df = pd.read_csv(argv[2], sep = '\t', header = 0, index_col = 0)
	# seperate feature data and response data 
	label_col_name = argv[5]
	X_train_data, y_train_data = train_data_df.drop(label_col_name, axis = 1).values, train_data_df[label_col_name].values
	X_test_data, y_test_data = test_data_df.drop(label_col_name, axis = 1).values, test_data_df[label_col_name].values
	# obtain names of all features 
	all_X_features = np.array(train_data_df.drop(label_col_name, axis = 1).columns) 
	
	## 2. Build L1 regularized classification/regression model and evaluate model performance 
	model_select, select_features, metric_select_train, metric_select_test = ttox_learning.regularize_by_l1(X_train_data, X_test_data, y_train_data, y_test_data, all_X_features, int(argv[6]), argv[7], int(argv[8]), seed_no = 0)	
	
	## 3. Build unregularized classification/regression model and evaluate model performance
	# use linear regression for regression tasks 
	if argv[7] == 'regression':
		model_name = 'Linear'
	# use logistic regression for classification tasks 
	if argv[7] == 'classification':
		model_name = 'Logistic'
	# build model   
	N_all_features, N_train_instances, N_test_instances, model_all, metric_all_test = ttox_learning.evaluate_model_performance(X_train_data, X_test_data, y_train_data, y_test_data, argv[7], model_name, seed_number = 0)
	
	## 4. Generate performance summary file  
	perf_summary = ttox_selection.generate_performance_summary(N_train_instances, N_test_instances, N_all_features, metric_all_test, select_features, len(select_features), metric_select_train, metric_select_test)
	perf_file = open(output_name + '_performance.txt', 'w')
	for ps in perf_summary:
		perf_file.write('%s\n' % ps)
	perf_file.close()

	## 5. Implements learned model to predict response for new instances
	if argv[3] != 'NA':
		pred_data_df = pd.read_csv(argv[3], sep = '\t', header = 0, index_col = 0)
		# use all features to predict 
		pred_label_all = ttox_learning.implement_prediciton_model(model_all, pred_data_df.values, argv[7])
		# use selected featuers to predict 
		pred_label_select = ttox_learning.implement_prediciton_model(model_select, pred_data_df.values, argv[7])
		# Aggregate predictions into one data frame (2 columns, 1: predictions from model using selected features, 2: predictions from model using all features) 
		pred_df = pd.DataFrame({'select_features_pred': pred_label_select,'all_features_pred': pred_label_all}, index = pred_data_df.index)
		pred_df.to_csv(output_name + '_prediction.tsv', sep = '\t', float_format = '%.5f')	

	return 1


## Call main function
main(sys.argv) 
