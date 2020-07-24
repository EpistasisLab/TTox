## created by Yun Hao @MooreLab 2019
## This script contains functions for visualizing feature selection results.


# Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


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


## This function visualizes density comparison of pairwise similarity scores among distinct classes  
def visualize_similarity_density_comparison(sim_dict, bg_sim, plot_file):
	## 0. Input arguments:
		# sim_dict: dictionary that contains the pairwise similarity scores among each class
		# bg_sim: array that contains the background pairwise similarity scores
		# plot_file: prefix of output plotting file 

	## 1. Make density plot 
	# specify figure and font size of density plot 
	plt.figure(figsize = (10, 10)) 
	plt.rc('font', size = 30)
	plt.rc('axes', titlesize = 30)
	plt.rc('axes', labelsize = 30)
	plt.rc('xtick', labelsize = 30)
	plt.rc('ytick', labelsize = 30)
	plt.rc('legend', fontsize = 15)
	# plot density of each class distribution 
	for key, value in sim_dict.items():  
		sns.distplot(value, hist = False, kde = True, kde_kws = {'linewidth': 3},  label = key)	
	# plot density of background distribution 
	sns.distplot(bg_sim, hist = False, kde = True, kde_kws = {'linewidth': 3},  label = 'background')
	# add labels and legends
	plt.xlabel('Jaccard Index')
	plt.ylabel('Density')
	plt.legend(loc = 'upper right')
	# save density plot
	plt.tight_layout()
	plt.savefig(plot_file + '_density.pdf')
	plt.close()

	## 2. Make boxplot
	# specify figure and font size of density plot 
	plt.figure(figsize = (15, 10))
	plt.rc('font', size = 30)
	plt.rc('axes', titlesize = 30)
	plt.rc('axes', labelsize = 30)
	plt.rc('xtick', labelsize = 15)
	plt.rc('ytick', labelsize = 15)
	plt.rc('legend', fontsize = 15)
	# include background distribution in the dictionary
	sim_dict['background'] = bg_sim
	class_labels, class_data = sim_dict.keys(), sim_dict.values()
	# separate words with '\n' to create new labels 
	class_labels1 = []
	for cl in class_labels:
		cl_s = cl.split(' ')
		class_labels1.append('\n'.join(cl_s))
	# plot distribution of each class in box 
	plt.boxplot(class_data, showfliers = False)
	plt.xticks(range(1, len(class_labels1) + 1), class_labels1)
	# add labels and legends
	plt.xlabel('Target class')
	plt.ylabel('Jaccard Index')
	# save density plot
	plt.tight_layout()
	plt.savefig(plot_file + '_boxplot.pdf')
	plt.close() 

	return 1
