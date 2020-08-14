## created by Yun Hao @MooreLab 2019
## This script contains functions for visualizing feature selection results.


# Module
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats


## This function makes boxplot comparing the performance of two sets of models that are different in a single hyperparameter.
def plot_hyperparameter_comparison_boxplot(perf_df, legend_label, output_fig):
	## 0. Input arguments 
		# perf_df: data frame containing model performance under different hyperparamter settings (row: dataset, column: hyperparameter)  
		# legend_label: label of legend that specifies compared hyperparamters 
		# output_fig: output figure file 
		
	## 1. Specify box colors 
	# obtain number of boxes
	N_col = perf_df.shape[1] 
	col_vec = np.tile(['red', 'blue'], int(N_col/2))

	## 2. Specify figure and font size 
	plt.figure(figsize = (N_col, 10))
	plt.rc('font', size = 30)
	plt.rc('axes', titlesize = 30)
	plt.rc('axes', labelsize = 30)
	plt.rc('xtick', labelsize = 30)
	plt.rc('ytick', labelsize = 30)
	plt.rc('legend', fontsize = 30)

	## 3. Make boxplot 
	ax = sns.boxplot(data = perf_df, palette = ['red', 'cyan'], showfliers = False)
	ax.set(xticklabels=[])
	ax.set_xlabel('Hyperparameter settings')
	ax.set_ylabel('Training performance (R squared)')
	ax.set_ylim(-1, 1.1)

	## 4. Add legend
	red_patch = mpatches.Patch(color = 'red', label = legend_label[0])
	cyan_patch = mpatches.Patch(color = 'cyan', label = legend_label[1])
	ax.legend(handles = [red_patch, cyan_patch], loc = 'upper left', ncol = 2)

	## 5. Save boxplot 
	plt.tight_layout()
	plt.savefig(output_fig)
	plt.close()

	return 1 

	
## This function visualizes model performance comparison under different hyperparameter settings. 
def visualize_hyperparameter_comparison(perf_df, output_folder):
	## 0. Input arguments: 
		# perf_df: data frame containing model performance under different hyperparamter settings (row: dataset, column: hyperparameter) 
		# output_folder: folder of output figures   

	## 1. Obtain detailed hyperparamter setting of each model
	# iterate by setting (column of performance data frame) 
	hp_list = []
	for pdc in perf_df.columns:
		pdc_s = pdc.split('_')
		hp_list.append({'ranking_method': pdc_s[1], 'implement_TURF': pdc_s[3], 'regression_method': pdc_s[7], 'tolerance': pdc_s[9], 'consistency': pdc_s[11]})
	# make data frame of hyperparamter setting of models
	hp_df = pd.DataFrame(hp_list)
	hp_df.index = perf_df.columns

	## 2. Use boxplot to visualize hyperparamter comparison 
	# compare ranking features by MultiSURF and MultiSURFstar 
	hp_df1 = hp_df.sort_values(by = ['implement_TURF', 'regression_method', 'tolerance', 'consistency', 'ranking_method'], axis = 0)
	perf_df1 = perf_df[hp_df1.index]
	ranking_method_compare = plot_hyperparameter_comparison_boxplot(perf_df1, ['MultiSURF', 'MultiSURFstar'], output_folder + '_ranking_method_compare.pdf')
	# compare implementing TURF and not implementing TURF
	hp_df2 = hp_df.sort_values(by = ['ranking_method', 'regression_method', 'tolerance', 'consistency', 'implement_TURF'], axis = 0)
	perf_df2 = perf_df[hp_df2.index]
	implement_TURF_compare = plot_hyperparameter_comparison_boxplot(perf_df2, ['No TURF', 'TURF'], output_folder + '_implement_TURF_compare.pdf')
	# compare building models by RandomForest and XGBoost
	hp_df3 = hp_df.sort_values(by = ['ranking_method', 'implement_TURF', 'tolerance', 'consistency', 'regression_method'], axis = 0)
	perf_df3 = perf_df[hp_df3.index]
	regression_method_compare = plot_hyperparameter_comparison_boxplot(perf_df3, ['RandomForest', 'XGBoost'], output_folder + '_regression_method_compare.pdf')
	# compare using tolerance of 20 and 50
	hp_df4 = hp_df.sort_values(by = ['ranking_method', 'implement_TURF', 'regression_method', 'consistency', 'tolerance'], axis = 0)
	perf_df4 = perf_df[hp_df4.index]
	tolerance_compare = plot_hyperparameter_comparison_boxplot(perf_df4, ['tolerance 20', 'tolerance 50'], output_folder + '_tolerance_compare.pdf')
	# compare using consistency threshold of 0.5 and 0.7
	hp_df5 = hp_df.sort_values(by = ['ranking_method', 'implement_TURF', 'regression_method', 'tolerance', 'consistency'], axis = 0)
	perf_df5 = perf_df[hp_df5.index]
	consistency_compare = plot_hyperparameter_comparison_boxplot(perf_df5, ['consistency 0.5', 'consistency 0.7'], output_folder + '_consistency_compare.pdf')

	return 1


## This function visualizes testing performance comparison between two models
def visualize_testing_performance_comparison(baseline_perf, model_perf, output_folder):
	## 0. Input arguments: 
		# baseline_perf: array of baseline performance using all features
		# model_perf: array of model performance using selected features
		# output_folder: folder of output figures   
	
	## 1. Specify figure and font size 
	plt.figure(figsize = (10, 10))
	plt.rc('font', size = 30)
	plt.rc('axes', titlesize = 20)
	plt.rc('axes', labelsize = 20)
	plt.rc('xtick', labelsize = 20)
	plt.rc('ytick', labelsize = 20)
	plt.rc('legend', fontsize = 20)
	
	## 2. Make scatter plot 
	plt.scatter(baseline_perf, model_perf, c = 'blue')
	plt.plot([0, 1], [0, 1], '-r')
	plt.xlim(-0.05, 1.05)
	plt.ylim(-0.05, 1.05)
	plt.xlabel('R squared by all features')
	plt.ylabel('R squared by selected features')

	## 3. Save boxplot 
	plt.tight_layout()
	plt.savefig(output_folder + '_testing_performance_compared.pdf')
	plt.close()

	return 1


## This function visualizes comparison of testing performance across different model classes. 
def visualize_class_performance_comparison(perf_df, perf_col, plot_file):
	## 0. Input arguements:
		# perf_df: data frame containing model performance (row: dataset, column 'class': protein function class of each model, column 'measure': measurement type of each model) 
		# perf_col: column of testing performance 
		# plot_file: predix of output plot files
	
	## 1. Visualizes comparison of testing performance across different protein function classes 
	# obtain all unique function classes  
	class_order = np.sort(perf_df['class'].unique())
	# iterate by function class 
	class_value_list = []	
	for i in range(0, len(class_order)):
		co = class_order[i] 
		# obtain testing performance of models that are built for the current function class 
		co_values = perf_df[perf_df['class'] == co][perf_col].values
		class_value_list.append(co_values)
	# perform KW test to examine whether the distribution of testing performance varies by function classes  
	compare_pv = stats.kruskal(*class_value_list)[1]
	# specify figure and font size  
	plt.figure(figsize = (24, 10))
	plt.rc('font', size = 30)
	plt.rc('axes', titlesize = 30)
	plt.rc('axes', labelsize = 30)
	plt.rc('xtick', labelsize = 20)
	plt.rc('ytick', labelsize = 20)
	plt.rc('legend', fontsize = 20)
	# make boxplot to show the distribution of testing performance across different function classes 
	ax = sns.boxplot(x = 'class', y = perf_col, data = perf_df, order = class_order)
	# add dashed line to show the median testing performance of all models 
	plt.axhline(y = perf_df[perf_col].median(), color = 'r', linestyle = '--', lw = 2, label = 'median of all')
	# add p value of KW test 
	plt.text(0, 1.05, 'P = ' + str(round(compare_pv, 2)) + '(KW text)', transform = ax.transAxes, size = 20)
	# add labels and legend 
	plt.xlabel('Target class')
	plt.ylabel('Testing performance')
	plt.legend(loc = 'upper right', bbox_to_anchor = (1, 1.1))
	# save boxplot 
	plt.tight_layout()
	plt.savefig(plot_file + '_by_class.pdf')
	plt.close()

	## 2. Visualizes comparison of testing performance across different measurement types 
	# obtain all measurement types 
	measure_order = np.sort(perf_df["measure"].unique())
	# iterate by measurement type
	measure_value_list = []
	for j in range(0, len(measure_order)):
		mo = measure_order[j]
		# obtain testing performance of models that are built for the current measurement type
		mo_values = perf_df[perf_df['measure'] == mo][perf_col].values
		measure_value_list.append(mo_values)
	# perform KW test to examine whether the distribution of testing performance varies by measurement type
	compare_pv = stats.kruskal(*measure_value_list)[1]
	# specify figure and font size 
	plt.figure(figsize = (10, 10))
	plt.rc('font', size = 30)
	plt.rc('axes', titlesize = 30)
	plt.rc('axes', labelsize = 30)
	plt.rc('xtick', labelsize = 30)
	plt.rc('ytick', labelsize = 30)
	plt.rc('legend', fontsize = 15)
	# make boxplot to show the distribution of testing performance across different measurement types
	ax = sns.boxplot(x = 'measure', y = perf_col, data = perf_df, order = measure_order)
	# add dashed line to show the median testing performance of all models
	plt.axhline(y = perf_df[perf_col].median(), color = 'r', linestyle = '--', lw = 2, label = 'median of all')
	# add p value of KW test 
	plt.text(0, 1.05, 'P = ' + str(round(compare_pv, 2)) + '(KW text)', transform = ax.transAxes, size = 20)
	# add labels and legend 
	plt.xlabel('Measurement')
	plt.ylabel('Testing performance')
	plt.legend(loc = 'upper right', bbox_to_anchor = (1, 1.1))
	# save boxplot 
	plt.tight_layout()
	plt.savefig(plot_file + '_by_measure.pdf')
	plt.close()

	return 1


## This function visualizes intergroup/intragroup comparison of pairwise similarity scores among distinct classes  
def visualize_class_similarity_comparison(sim_df, sim_pv, class_type, plot_file):
	## 0. Input arguments:
		# sim_df: data frame that contains the pairwise similarity scores among each class (column 'class': class of each model, column 'group': group within the class, column 'similarity')
		# sim_pv: dictionary that contains the p value of intragroup/intergroup comparison (key: class name, value: p-value)
		# class_type: type of class to be compared, shown as x-axis label 
		# plot_file: prefix of output plot file 

	## 1. Identify classes with significant p-values in the comparison
	# obtain all unique classes 
	class_labels = np.sort(list(sim_pv.keys()))
	# compute the Bonferroni correction p value threshold for comparing all classes  
	pv_cut = 0.05/len(class_labels)
	# iterate by class
	class_labels1 = []
	greater_sig_x = []
	greater_sig_y = []
	max_value = np.quantile(sim_df.similarity.values, 0.995)
	for i in range(0, len(class_labels)):
		cl = class_labels[i]
		# customize class names to show on the plot 
		cl_s = cl.split(' ')
		class_labels1.append('\n'.join(cl_s))
		# identify classes with significant p-values 
		if sim_pv[cl] < pv_cut:
			greater_sig_x.append(i)
			greater_sig_y.append(max_value)
	
	## 2. Specify figure and font size of density plot 
	plt.figure(figsize = (15, 10))
	plt.rc('font', size = 30)
	plt.rc('axes', titlesize = 30)
	plt.rc('axes', labelsize = 30)
	plt.rc('xtick', labelsize = 15)
	plt.rc('ytick', labelsize = 15)
	plt.rc('legend', fontsize = 15)
	
	## 3. Visualizes intergroup/intragroup comparison of pairwise similarity scores among distinct classes
	sns.boxplot(x = 'class', y = 'similarity', hue = 'group', data = sim_df, order = class_labels, showfliers = False)	
	plt.xticks(range(0, len(class_labels1)), class_labels1)
	# add marker to show classes with significant p-values 
	if len(greater_sig_x) > 0:
		plt.plot(greater_sig_x, greater_sig_y, marker = '*', color = 'r', linestyle = 'None', markersize = 15, label = 'intragroup > intergroup (P < 0.05)')
	# add labels and legend
	plt.xlabel(class_type)
	plt.ylabel('Feature similarity')
	plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.1), ncol = 3)
	
	## 4. Save density plot
	plt.tight_layout()
	plt.savefig(plot_file + '_boxplot.pdf')
	plt.close() 

	return 1
