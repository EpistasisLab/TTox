# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script compares and visulizes performance of distinct toxicity prediction models 

# module
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/')
import ttox_learning
import ttox_plot


def main(argv):
	## 0. Input arguments 
		# argv 1: input file that contains optimal hyperparamter file names  
		# argv 2: input file that contains dataset info (number of positive/negative samples, etc)
		# argv 3: number of times to repeat the model training process, the average prediction from all models will be used to evaluate model performance  
		# argv 4: output file that contains computed AUC and confidence interval metrics  
		# argv 5: plot file that contains ROC curve of models  
		# argv 6: prefix of plot file contains scatter plots comparing AUROC   
	
	## 1. Read in file names that store optimal hyperparameter values 
	optimal_hps = np.loadtxt(argv[1], dtype = 'str', delimiter = '\n')
		
	## 2. Compute confidence interval of testing performance for model built upon distinct feature feature types
	perf_dict = {}
	ae_all_features = []
	ae_select_features = []
	# iterate by adverse event dataset 
	for oh in optimal_hps:
		oh_perf_dict = {}
		oh_roc = []
		# obtain structure feature type and adverse event name 
		oh_s = oh.split('/')
		str_feature_type = oh_s[2]
		ae_name = oh_s[3].split('_select')[0]   

		# obtain hyperparameter values of optimal setting 
		oh_lines = np.loadtxt(oh, dtype = 'str', delimiter = '\n')
		oh_values = []
		for ol in oh_lines:
			oh_values.append(ol.split(': ')[1])
		oh_name = 'mc_' + oh_values[0] + '_fd_' + oh_values[1] + '_fr_' + oh_values[2] + '_tf_' + oh_values[3] + '_pc_' + oh_values[4] + '_md_' + oh_values[6] + '_tl_' + oh_values[7] + '_cs_' + oh_values[8] + '_nr_' + oh_values[9]
		
		# read in compound structure-adverse event training data 
		str_train_file = 'data/compound_structure_all_adverse_event_data/compound_target_' + str_feature_type + '_' + ae_name + '_whole_data.tsv_train.tsv'
		str_train_df = pd.read_csv(str_train_file, sep = '\t', header = 0, index_col = 0)	
		str_X_train, str_y_train = str_train_df.drop('toxicity_label', axis = 1).values, str_train_df['toxicity_label'].values
		# read in compound structure-adverse event testing data 
		str_test_file = 'data/compound_structure_all_adverse_event_data/compound_target_' + str_feature_type + '_' + ae_name + '_whole_data.tsv_test.tsv'
		str_test_df = pd.read_csv(str_test_file, sep = '\t', header = 0, index_col = 0)
		str_X_test, str_y_test = str_test_df.drop('toxicity_label', axis = 1).values, str_test_df['toxicity_label'].values
		# compute confidence interval of testing performance for model built upon structure-adverse event dataset 
		oh_perf_dict['structure_model_auc'], oh_perf_dict['structure_model_auc_ci'], oh_perf_dict['structure_model_bootstrap_prop'], structure_fpr, structure_tpr = ttox_learning.compute_classification_model_auc_ci(str_X_train, str_X_test, str_y_train, str_y_test, oh_values[6], n_repeat = int(argv[3]))
		oh_perf_dict['N_structure_features'] = str_X_train.shape[1]

		# read in compound target-adverse event training data 	
		tar_train_file = 'data/compound_target_all_adverse_event_data/compound_target_' + str_feature_type + '_mc_' + oh_values[0] + '_' + ae_name + '_whole_data.tsv_train.tsv'	
		tar_train_df = pd.read_csv(tar_train_file, sep = '\t', header = 0, index_col = 0) 
		tar_X_train, tar_y_train = tar_train_df.drop('toxicity_label', axis = 1).values, tar_train_df['toxicity_label'].values
		# read in compound target-adverse event testing data  
		tar_test_file = 'data/compound_target_all_adverse_event_data/compound_target_' + str_feature_type + '_mc_' + oh_values[0] + '_' + ae_name + '_whole_data.tsv_test.tsv' 
		tar_test_df = pd.read_csv(tar_test_file, sep = '\t', header = 0, index_col = 0)
		tar_X_test, tar_y_test = tar_test_df.drop('toxicity_label', axis = 1).values, tar_test_df['toxicity_label'].values
		# compute confidence interval of testing performance for model built upon target-adverse event dataset 
		oh_perf_dict['target_all_model_auc'], oh_perf_dict['target_all_model_auc_ci'], oh_perf_dict['target_all_model_bootstrap_prop'], all_tar_fpr, all_tar_tpr = ttox_learning.compute_classification_model_auc_ci(tar_X_train, tar_X_test, tar_y_train, tar_y_test, oh_values[6], n_repeat = int(argv[3]))	
		oh_perf_dict['N_all_target_features'] = tar_X_train.shape[1]
		ae_all_features.append(','.join(tar_train_df.drop('toxicity_label', axis = 1).columns)) 

		# obtain selected features of optimal feature selection pipeline 
		tar_perf_file = '/'.join(oh_s[0:3]) + '/' + ae_name + '/compound_target_' + str_feature_type + '_mc_' + oh_values[0] + '_' + ae_name + '_whole_data.tsv_' + oh_name + '_performance.txt'
		tar_perf = np.loadtxt(tar_perf_file, dtype = 'str', delimiter = '\n')
		tar_features_vec = tar_perf[4].split(': ')[1]
		ae_select_features.append(tar_features_vec)
		tar_features = tar_features_vec.split(',')
		# compute confidence interval of testing performance for optimal feature selection model built upon target-adverse event dataset 
		oh_perf_dict['target_select_model_auc'], oh_perf_dict['target_select_model_auc_ci'], oh_perf_dict['target_select_model_bootstrap_prop'], select_tar_fpr, select_tar_tpr = ttox_learning.compute_classification_model_auc_ci(tar_train_df[tar_features].values, tar_test_df[tar_features].values, tar_y_train, tar_y_test, oh_values[6], n_repeat = int(argv[3]))
		oh_perf_dict['N_select_target_features'] = len(tar_features)
		perf_dict[ae_name] = oh_perf_dict

		# make list that stores the performance metrics of models to be compared   
		select_tar_label = str(oh_perf_dict['N_select_target_features']) + ' selected targets\n(AUC=' + str(round(oh_perf_dict['target_select_model_auc'], 2)) + '±' + str(round(oh_perf_dict['target_select_model_auc_ci'], 2)) + ')'
		oh_roc.append((select_tar_fpr, select_tar_tpr, select_tar_label))
#		all_tar_label = str(oh_perf_dict['N_all_target_features']) + ' targets\n(AUC = ' + str(round(oh_perf_dict['target_all_model_auc'], 2)) + '±' + str(round(oh_perf_dict['target_all_model_auc_ci'], 2)) + ')'
#		oh_roc.append((all_tar_fpr, all_tar_tpr, all_tar_label))
		if str_feature_type == 'descriptor_all':
			structure_label = str(oh_perf_dict['N_structure_features']) + ' molecular descriptors\n(AUC=' + str(round(oh_perf_dict['structure_model_auc'], 2)) + '±' + str(round(oh_perf_dict['structure_model_auc_ci'], 2)) + ')'
		if str_feature_type == 'fingerprint_maccs':
			structure_label = str(oh_perf_dict['N_structure_features']) + ' maccs fingerprints\n(AUC=' + str(round(oh_perf_dict['structure_model_auc'], 2)) + '±' + str(round(oh_perf_dict['structure_model_auc_ci'], 2)) + ')'
		oh_roc.append((structure_fpr, structure_tpr, structure_label))
		roc_compare = ttox_plot.visualize_model_roc_comparison(oh_roc, ae_name, argv[5] + '_' + str(argv[3]) + '_' + ae_name)	
	
	## 3. Output performance metrics and selected features in data frame format 
	# performance metrics
	perf_df = pd.DataFrame(perf_dict).T
	perf_df.to_csv(argv[4] + '_' + str(argv[3]) + '_testing_performance_summary_auc_ci.tsv', sep = '\t', float_format = '%.5f')
	# all features 
	all_feat_df = pd.DataFrame({'adverse_event': perf_df.index, 'all_features': ae_all_features}) 
	all_feat_df.to_csv(argv[4] + '_all_features.tsv', sep = '\t', index = False, header = True)
	# selected features
	select_feat_df = pd.DataFrame({'adverse_event': perf_df.index, 'select_features': ae_select_features})	
	select_feat_df.to_csv(argv[4] + '_select_features.tsv', sep = '\t', index = False, header = True)

	## 4. Compare and visualize comparison of testing performance across different feature types
	# structure features vs selected target features  
	structure_select_compare = ttox_plot.plot_comparison_scatter(perf_df['structure_model_auc'].values, perf_df['target_select_model_auc'].values, [0.5, 0.75], [], 'AUROC by structure features', 'AUROC by selected targets', argv[6] + '_structure_select.pdf', True)
	# all target features vs selected target features 
	all_select_compare = ttox_plot.plot_comparison_scatter(perf_df['target_all_model_auc'].values, perf_df['target_select_model_auc'].values, [0.5, 0.75], [], 'AUROC by all targets', 'AUROC by selected targets', argv[6] + '_all_select.pdf', True)

	## 5. Visualize correlation between ratio of positives to negatives and width of CI for AUROC
	# read in data frame that contains dataset information 
	info_df = pd.read_csv(argv[2], sep = '\t')
	perf_info_df = pd.merge(perf_df, info_df, left_index = True, right_on = 'Group')
	# ratio of positives to negatives vs width of CI for AUROC
	pos_to_neg_ratio = perf_info_df.N_positive_samples.values/perf_info_df.N_negative_samples.values
	auc_ci_width = 2 * perf_info_df.target_select_model_auc_ci.values
	ratio_width_cor = ttox_plot.visualize_correlation_scatter(pos_to_neg_ratio, auc_ci_width, '#Positives/#Negatives', 'Width of 95% CI for AUROC', argv[6] + '_p_to_n_ratio_vs_auc_ci_width.pdf', True)
	
	return 1


## call main function
main(sys.argv)
