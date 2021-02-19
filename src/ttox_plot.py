# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains functions for visualizing feature selection results.


# Module
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
sys.path.insert(0, 'src/')
import ttox_selection

## This function visualizes performance comparison between two sets of models that are different in a single hyperparameter
def plot_hyperparameter_comparison_boxplot(perf_df, task, legend_label, output_fig):
	## 0. Input arguments 
		# perf_df: data frame containing model performance under different hyperparamter settings (row: dataset, column: hyperparameter)  
		# task: type of supervised learning task: 'regression' or 'classification' 
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

	## 3. Visualize performance comparison between two sets of models that are different in a single hyperparameter
	# make boxplot comparing the performance of two sets of models 
	ax = sns.boxplot(data = perf_df, palette = ['red', 'cyan'], showfliers = False, notch = True)
	ax.set(xticklabels=[])
	# set axis labels and ranges for regression tasks  
	if task == 'regression':
		ax.set_xlabel('Hyperparameter settings')
		ax.set_ylabel('Training performance (R squared)')
		ax.set_ylim(-1, 1.1)
	# set axis labels and ranges for classification tasks  
	if task == 'classification':
		ax.set_xlabel('Hyperparameter settings')
		ax.set_ylabel('Training performance (AUC)')
		ax.set_ylim(0.5, 1.1)
	# add legend showing the box colors 
	red_patch = mpatches.Patch(color = 'red', label = legend_label[0])
	cyan_patch = mpatches.Patch(color = 'cyan', label = legend_label[1])
	ax.legend(handles = [red_patch, cyan_patch], loc = 'upper left', ncol = 2, frameon = False)

	## 4. Save boxplot 
	plt.tight_layout()
	plt.savefig(output_fig)
	plt.close()

	return 1 

	
## This function visualizes model performance comparison under different hyperparameter settings
def visualize_hyperparameter_comparison(perf_df, task, output_folder):
	## 0. Input arguments: 
		# perf_df: data frame containing model performance under different hyperparamter settings (row: dataset, column: hyperparameter) 
		# task:	type of supervised learning task: 'regression' or 'classification' 
		# output_folder: folder of output figures   

	## 1. Obtain detailed hyperparamter setting of each model
	hp_list = []
	# iterate by setting (column of performance data frame) 
	for pdc in perf_df.columns:
		# obtain the detailed hyperparameter value of each setting
		pdc_s = pdc.split('_')
		hp_list.append({'ranking_method': pdc_s[1], 'implement_TURF': pdc_s[3], 'learning_method': pdc_s[7], 'tolerance': pdc_s[9], 'consistency': pdc_s[11]})
	# names of hyperparameter value pairs to be compared 
	hp_name = [['MultiSURF', 'MultiSURFstar'], ['No TURF', 'TURF'], ['RandomForest', 'XGBoost'], ['tolerance 20', 'tolerance 50'], ['consistency 0.5', 'consistency 0.7']]
	# make data frame of hyperparamter setting of models
	hp_df = pd.DataFrame(hp_list)
	hp_df.index = perf_df.columns
	
	## 2. Visualize hyperparamter comparison by boxplot 
	# compare ranking features by MultiSURF and MultiSURFstar 
	for i in range(0, len(hp_df.columns)): 
		# iterate by hyperparameter
		hdc = hp_df.columns[i]
		# sort hyperparameter settings by order of other hyperparameters 
		compare_factor = hp_df.columns.tolist()
		compare_factor.remove(hdc)
		compare_factor.append(hdc)
		hp_df1 = hp_df.sort_values(by = compare_factor, axis = 0)
		perf_df1 = perf_df[hp_df1.index]
		# visualize the pairwise comparison  	
		ranking_method_compare = plot_hyperparameter_comparison_boxplot(perf_df1, task, hp_name[i], output_folder + '_' + hdc + '_compare.pdf')

	return 1


## This function visualizes performance comparison between two sets of models  
def plot_comparison_scatter(x_metric, y_metric, xy_lim, cut_list, x_label, y_label, plot_file, show_pv = True):
	## 0. Input arguments: 
		# x_metric: metric values of set 1, to be plotted on the x axis  
		# y_metric: metric values of set 2, to be plotted on the y axis  
		# xy_lim: arrays that contains plotting range of x and y axis
		# cut_list: arrays that contains value thresholds 
		# x_label: label of x axis 
		# y_label: label of y axis 
		# plot_file: file name of output figures  
		# show_pv: whether to show p value comparing the x and y axis values 	

	## 1. Specify plotting parameters 
	# x and y axis range 
	xy_lim_lower = xy_lim[0] 
	xy_lim_upper = xy_lim[1]
	xy_range = xy_lim_upper - xy_lim_lower
	# y coordiante to label p-value 
	pv_y_margin = 0.02 * xy_range + 0.00125
	pv_y = xy_lim_upper + pv_y_margin
	# x coordiantes to label p-value 
	pv_x_margin = 0.0089 * xy_range + 0.01
	pv_x1 = xy_lim_lower + pv_x_margin
	pv_x2 = xy_lim_upper + pv_x_margin
	# colors to label p-value thresholds in 
	col_vec = ['magenta', 'cyan', 'green']

	## 2. Specify figure and font size 
	plt.figure(figsize = (6, 6))
	plt.rc('font', size = 22)
	plt.rc('axes', titlesize = 25)
	plt.rc('axes', labelsize = 22)
	plt.rc('xtick', labelsize = 22)
	plt.rc('ytick', labelsize = 22)
	plt.rc('legend', fontsize = 22)

	## 3. Visualize performance comparison between two sets of models by scatter plot
	# make scatter plot
	plt.scatter(x_metric, y_metric, c = 'blue')
	plt.plot(xy_lim, xy_lim, '-r')
	# perform wilcoxon test to examine whether y_metric is greater than x_metric
	if show_pv == True:
		pv = ttox_selection.compare_sample_means_by_one_sided_test(y_metric, x_metric, False, True)
		text_col = 'black'
		if pv < 0.05:
			text_col = 'red'
		plt.text(pv_x1, pv_y, 'P=' + np.format_float_scientific(pv, precision = 1) + ' (y > x)', size = 22, c = text_col)
	# iterate by threshold, compare subsets of performance metrics that are above the threshold  
	for i in range(0, len(cut_list)):
		# obtain model subsets of which both metrics pass the threshold 
		cl = cut_list[i]
		y_id = y_metric > cl
		xy_id = (x_metric > cl) * y_id
		# perform wilcoxon test to examine whether the subset of y_metric is greater than x_metric
		cl_pv = ttox_selection.compare_sample_means_by_one_sided_test(y_metric[xy_id], x_metric[xy_id], False, True)
		# add labels to show the comparison under the threshold, label the p-value
		plt.plot([cl, cl], [cl, xy_lim_upper], '--', c = col_vec[i])
		plt.plot([cl, xy_lim_upper], [cl, cl], '--', c = col_vec[i])
		plt.text(cl + pv_x_margin, pv_y, 'P=' + np.format_float_scientific(cl_pv, precision = 2), size = 20, c = col_vec[i])
		plt.text(pv_x2, cl, str(cl), size = 20, c = col_vec[i])
	# set x and y axis  
	plt.xlim(xy_lim)
	plt.ylim(xy_lim)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	
	## 4. Save boxplot 
	plt.tight_layout()
	plt.savefig(plot_file)
	plt.close()

	return 1


## This function visualizes testing performance comparison between models.
def visualize_testing_performance_comparison(file_df, plot_task, select_rows, output_folder, l1_file_name, output_folder_f1):
	## 0. Input arguments:
		# file_df: data frame that contains performance file names  
		# plot_task: type of supervised learning task for which models are built: 'regression' or 'classification' 
		# select_rows: subset of models that are selected for comparison (one model per target)  
		# output_folder: folder of output figures that visualize comparison between selected features and all features  
		# l1_file_name: name of file that contains performance file names of L1-regularized models  
		# output_folder_f1: folder of output figures that visualize comparison between selected features and L1-selected features  
		
	## 1. Specift plotting hyperparameters 
	# hyperparameters for regression tasks  
	if plot_task == 'regression':
		plot_metric = ['r2']
		metric_name = ['R^2']
		plot_range = [-0.05, 1.05]
		plot_cut = [0.5, 0.75]
	# hyperparameters for classification tasks  
	if plot_task == 'classification':
		plot_metric = ['auc', 'bac', 'f1']
		metric_name = ['AUROC', 'BA', 'F1']
		plot_range = [0.48, 1.02]
#		plot_cut = [0.7, 0.8, 0.9]
		plot_cut = []
	# read in the data frame that contains performance of L1 regulariztion models (if applicable) 
	if l1_file_name != 'NA':
		l1_file_df = pd.read_csv(l1_file_name, sep = '\t', header = 0, index_col = 0)
		
	## 2. Visualize comparison of model performance under different metrics   
	# iterate by performance metric
	for lpm in range(0, len(plot_metric)):
		pm = plot_metric[lpm]
		mn = metric_name[lpm]
		# obtain performance metric of models built on selected features (to be plotted on y axis)
		model_file = file_df.loc['select_test', pm]
		model_df = pd.read_csv(model_file, sep = '\t', header = 0, index_col = 0)
		perf_col = model_df.columns[0]
		model_perf = model_df.loc[select_rows, perf_col].values
		# obtain performance metric of models built on all features (to be plotted on x axis)
		baseline_file = file_df.loc['all_test', pm]
		baseline_df = pd.read_csv(baseline_file, sep = '\t', header = 0, index_col = 0)
		baseline_perf = baseline_df.loc[select_rows, perf_col].values
		# visualize comparison between model using selected features and model using all features 
		pm_xlabel = mn + ' before selection'
		pm_ylabel = mn + ' after ReBATE selection'
		pm_file = output_folder + '_all_testing_performance_compared_' + pm + '.pdf' 
		baseline_compare = plot_comparison_scatter(baseline_perf, model_perf, plot_range, plot_cut, pm_xlabel, pm_ylabel, pm_file)
		# check if L1 regulariztion models are also included in the comparison 
		if l1_file_name != 'NA':
			# obtain performance metric of models built with L1 regulariztion (to be plotted on x axis)
			l1_file = l1_file_df.loc['select_test', pm]
			l1_basedline_df = pd.read_csv(l1_file, sep = '\t', header = 0, index_col = 0)
			l1_col = l1_basedline_df.columns[0]
			l1_baseline_perf = l1_basedline_df.loc[select_rows, l1_col].values 
			# visualize comparison between model using selected features and model using L1 regularization 
			f1_s = output_folder_f1.split('_')
			l1_method = f1_s[len(f1_s) - 1]
			if l1_method == 'randomforest':
				pm_xlabel_l1 = mn + ' after L1 selection'
				pm_ylabel_l1 = mn + ' after ReBATE selection'
			if l1_method == 'lasso':
				pm_xlabel_l1 = mn + ' by L1+logistic regression'
				pm_ylabel_l1 = mn + ' by ReBATE+Randomforest'
			pm_l1_file = output_folder_f1 + '_l1_testing_performance_compared_' + pm + '.pdf'
			l1_baseline_compare = plot_comparison_scatter(l1_baseline_perf, model_perf, plot_range, plot_cut, pm_xlabel_l1, pm_ylabel_l1, pm_l1_file)
	
	return 1


## This function visualizes feature number comparison between models. 
def visualize_feature_number_comparison(select_number_df, l1_number_df, select_rows, output_folder):
	## 0. Input arguments:
		# select_number_df: data frame that contains number of selected features  
		# l1_number_df: data frame that contains number of L1 selected features 
		# select_rows: subset of models that are selected for comparison (one model per target)  
		# output_folder: folder of output figure
			
	## 1. Obtain numbers of selected features for specified models 
	l1_number = l1_number_df.loc[select_rows,].iloc[: ,0].values
	select_number = select_number_df.loc[select_rows,].iloc[: ,0].values
	number_df = pd.DataFrame({'ReBATE': select_number, 'L1': l1_number})	

	## 2. Specify figure and font size 
	plt.figure(figsize = (3.5, 6))
	plt.rc('font', size = 22)
	plt.rc('axes', titlesize = 25)
	plt.rc('axes', labelsize = 22)
	plt.rc('xtick', labelsize = 22)
	plt.rc('ytick', labelsize = 22)
	plt.rc('legend', fontsize = 22)

	## 3. Visualize feature number comparison by boxplot
	ax = sns.boxplot(data = number_df, showfliers = False, notch = True, color = 'lightgray')
	# set label, title
	ax.set(ylabel = 'Number of selected features')

	## 4. Save boxplot 
	plt.tight_layout()
	plt.savefig(output_folder + '_feature_number_compared.pdf')
	plt.close()

	return 1		


## This function visualizes comparison of testing performance across different model classes. 
def visualize_group_performance_comparison(perf_df, perf_col, group_col, group_type, group_order, plot_file):
	## 0. Input arguements:
		# perf_df: data frame containing model performance (row: dataset, column 'class': protein function class of each model, column 'measure': measurement type of each model) 
		# perf_col: column of testing performance 
		# group_col: name of column that contains class info of target
		# group_type: type of target classes to be compared, shown as x-axis ticks 
		# group_order: order of target classes to be shown as x-axis ticks 
		# plot_file: prefix of output plot files
	
	## 1. Obtain hyperparameters for plotting: class labels and comparison P values  
	# iterate by function class 
	group_value_list = []	
	group_label = []
	for i in range(0, len(group_order)):
		go = group_order[i] 
		# customize class names to show on the plot 
		go_s = go.split(' ')
		group_label.append('\n'.join(go_s))
		# obtain testing performance of models that are built for the current function class 
		go_values = perf_df[perf_df[group_col] == go][perf_col].values
		group_value_list.append(go_values)
	# perform KW test to examine whether the distribution of testing performance varies by function classes  
	compare_pv = stats.kruskal(*group_value_list)[1]

	## 2. Specify figure and font size  
	plt.figure(figsize = (len(group_order)*1.25, 6))
	plt.rc('font', size = 17)
	plt.rc('axes', titlesize = 25)
	plt.rc('axes', labelsize = 17)
	plt.rc('xtick', labelsize = 17)
	plt.rc('ytick', labelsize = 17)
	plt.rc('legend', fontsize = 17)
	
	## 3. Visualizes comparison of testing performance across different protein function classes  
	# make boxplot to show the distribution of testing performance across different function classes 
	ax = sns.boxplot(x = group_col, y = perf_col, data = perf_df, order = group_order, showfliers = False, notch = True)
	plt.xticks(range(0, len(group_label)), group_label)
	# add dashed line to show the median testing performance of all models 
	plt.axhline(y = perf_df[perf_col].median(), color = 'r', linestyle = '--', lw = 2, label = 'median of all')
	# add p value of KW test
	plt.text(0, 1.02, 'P=' + str(round(compare_pv, 3)) + ' (Kruskal–Wallis test)', transform = ax.transAxes, size = 17)
	# add labels and legend 
	plt.xlabel(group_type)
	plt.ylabel('Model AUROC')
	plt.legend(loc = 'lower left', frameon = False)

	## 4. Save boxplot 
	plt.tight_layout()
	sns.despine()
	plt.savefig(plot_file + '_by_' + group_col + '.pdf')
	plt.close()

	return 1


## This function visualizes class composition of target proteins   
def visualize_group_proportion(target_group_df, group_col, group_type, group_order, plot_file):
	## 0. Input arguments: 
		# target_group_df: data frame that contains class info of target 
		# group_col: name of column that contains class info of target
		# group_type: type of target classes to be compared, shown as x-axis ticks 
		# group_order: order of target classes to be shown as x-axis ticks
		# plot_file: prefix of output plot files

	## 1. Obtain hyperparameters for plotting: group distribution, group labels 
	# obtain class distribution in the specified order
	target_group_count = target_group_df.groupby(group_col)['target'].nunique() 
	count_df = target_group_count.loc[group_order,]
	group_size = count_df.values
	N_all = np.sum(group_size) 
	# obtain customized class labels to be shown as as x-axis ticks 
	group_labels = count_df.index
	group_labels1 = []
	for i in range(0, len(group_labels)):
		gl = group_labels[i]
		# substitute ' ' in the name with '\n'   
		gl_s = gl.split(' ')
		group_labels1.append('\n'.join(gl_s))

	## 2. Specify figure and font size
	plt.figure(figsize = (8, 6))
	plt.rc('font', size = 22)
	plt.rc('axes', titlesize = 25)
	plt.rc('axes', labelsize = 22)
	plt.rc('xtick', labelsize = 22)
	plt.rc('ytick', labelsize = 22)
	plt.rc('legend', fontsize = 22)
	
	## 3. Visualize class composition   
	# make pie plot showing class composition  
	plt.pie(count_df.values, labels = group_labels1, autopct = '%1.1f%%', shadow = True, startangle = 25)
	plt.title(group_type + ' (N=' + str(N_all) + ')')
	# add x axis label
	plt.axis('equal')

	## 4. Save pie chart
	plt.tight_layout()
	plt.savefig(plot_file + '_by_' + group_col + '.pdf')
	plt.close()

	return 1


## This function visualizes comparison of intergroup and intragroup feature similarity among classes  
def visualize_group_feature_similarity_comparison(sim_df, sim_pv, group_col, group_type, group_order, plot_file):
	## 0. Input arguments:
		# sim_df: data frame that contains the pairwise similarity scores among each class (column 'class': class of each model, column 'group': group within the class, column 'similarity')
		# sim_pv: dictionary that contains the p value of intragroup/intergroup comparison (key: class name, value: p-value)
		# group_col: name of column that contains class info of target
		# group_type: type of target classes to be compared, shown as x-axis ticks
		# group_order: order of target classes to be shown as x-axis ticks
		# plot_file: prefix of output plot file 

	## 1. Obtain hyperparameters for plotting: class labels, comparison p values
	# obtain class labels to be shown as x-axis ticks
	group_labels = []
	group_pv = []
	for i in range(0, len(group_order)):
		gl = group_order[i]
		# customize class names  
		gl_s = gl.split(' ')
		group_labels.append('\n'.join(gl_s))
		group_pv.append(sim_pv[gl])
	# perform multiple testing correction among different classes  
	group_reject, group_fdr, _, _ = multipletests(group_pv, method = 'fdr_bh')
	# identify classes with significant p-values in the comparison 
	sig_x_loc = []
	sig_y_loc = []	
	for gr in range(0, len(group_reject)):
		if group_reject[gr]:
			sig_x_loc.append(gr)
			sig_y_loc.append(0)

	## 2. Specify figure and font size 
	plt.figure(figsize = (10, 6))
	plt.rc('font', size = 20)
	plt.rc('axes', titlesize = 20)
	plt.rc('axes', labelsize = 20)
	plt.rc('xtick', labelsize = 20)
	plt.rc('ytick', labelsize = 20)
	plt.rc('legend', fontsize = 20)
	
	## 3. Visualizes comparison of intergroup and intragroup similarity among classes
	# make boxplot showing the intergroup/intragroup comparison of pairwise similarity scores
	ax = sns.boxplot(x = group_col, y = 'similarity', hue = 'group', data = sim_df, order = group_order, showfliers = False, notch = True)	
	ax.set_ylim([-0.02, 0.4])
	ax.get_legend().remove()
	# show class names as x ticks 
	plt.xticks(range(0, len(group_labels)), group_labels)
	# set the color of boxes (grouped by classes) 
	col_set = plt.rcParams['axes.prop_cycle'].by_key()['color']
	for i in range(0, len(group_order)):
		mybox = ax.artists[2*i]
		mybox.set_facecolor(col_set[i])
		mybox = ax.artists[2*i+1]
		mybox.set_facecolor(col_set[i])		
	# add marker to show classes with significant p-values 
	if len(sig_x_loc) > 0:
		plt.plot(sig_x_loc, sig_y_loc, marker = '*', color = 'r', linestyle = 'None', markersize = 15)
	# add axis labels 
	plt.xlabel(group_type)
	plt.ylabel('Feature similarity')
	# add legend showing the significance threshold of markers  
	legend_elements = [Line2D([0], [0], marker = '*', color = 'w', label = 'intraclass,left > interclass,right (FDR<0.05)', markerfacecolor = 'r', markersize = 15)]
	plt.legend(handles = legend_elements, loc = 'upper right', frameon = False)
	
	## 4. Save boxplot
	plt.tight_layout()
	sns.despine()
	plt.savefig(plot_file + '_by_' + group_col + '_boxplot.pdf')
	plt.close() 

	return 1


## This function visualizes performance comparison of multiple classifiers by ROC plot 
def visualize_model_roc_comparison(roc_list, roc_title, plot_file):
	## 0. Input arguments:
		# roc_list: list that contains paired TPRs and FPRs to be plotted 
		# roc_title: title of ROC plot 
		# plot_file: prefix of output plot file  

	## 1. Specify figure and font size
	plt.figure(figsize = (6, 6))
	plt.rc('font', size = 20)
	plt.rc('axes', titlesize = 20)
	plt.rc('axes', labelsize = 20)
	plt.rc('xtick', labelsize = 15)
	plt.rc('ytick', labelsize = 15)
	plt.rc('legend', fontsize = 15)

	## 2. Visualizes performance comparison of multiple classifiers 
	# plot ROC curves of multiple classifiers 
	for rl in roc_list:
		rl_fpr, rl_tpr, rl_label = rl
		plt.plot(rl_fpr, rl_tpr, lw = 2, label = rl_label)
	# plot baseline curve of a random classifier (diagonal line)
	plt.plot([0, 1], [0, 1], color = 'grey', lw = 1, linestyle='--')
	# set axis ranges
	plt.xlim([-0.01, 1])
	plt.ylim([0, 1.01])
	# set axis labels
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	# set main title
	roc_title = ' '.join(roc_title.split('_'))
	plt.title(roc_title)
	# set legend showing the labels of classifiers 
	plt.legend(loc = 'lower right', frameon = False)

	## 3. Save ROC plot
	plt.tight_layout()
	plt.savefig(plot_file + '_roc_curve.pdf')
	plt.close()

	return 1


## This function visualizes correlation between two vectors by scatter plot
def visualize_correlation_scatter(x_value, y_value, x_label, y_label, plot_file, show_pv = True):
	## 0. Input arguments: 
		# x_value: values of set 1, to be plotted on the x axis  
		# y_value: values of set 2, to be plotted on the y axis  
		# x_label: label of x axis 
		# y_label: label of y axis 
		# plot_file: file name of output figures  
		# show_pv: whether to show p value comparing the x and y axis values    

	## 1. Specify plotting parameters 
	# y axis range 
	y_lower = np.min(y_value)
	y_upper = np.max(y_value)
	y_range = y_upper - y_lower
	# y coordiante to label p-value 
	pv_y_margin = y_range/15
	pv_y = y_upper + pv_y_margin
	# x coordiantes to label p-value 
	pv_x = np.min(x_value)

	## 2. Specify figure and font size 
	plt.figure(figsize = (6, 6))
	plt.rc('font', size = 22)
	plt.rc('axes', titlesize = 25)
	plt.rc('axes', labelsize = 22)
	plt.rc('xtick', labelsize = 22)
	plt.rc('ytick', labelsize = 22)
	plt.rc('legend', fontsize = 22)

	## 3. Visualiz correlation between two vectors by scatter plot
	# make scatter plot
	plt.scatter(x_value, y_value, c = 'grey')
	# compute Spearman correlation coefficient and p-value
	if show_pv == True:
		s_cor, pv = stats.spearmanr(x_value, y_value)
		text_col = 'black'
		if pv < 0.05:
			text_col = 'red'
		pv_text = 'ρ=' + str(round(s_cor, 2)) + ', P=' + np.format_float_scientific(pv, precision = 1)
		plt.text(pv_x, pv_y, pv_text, size = 22, c = text_col)
	# set x and y axis  
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	## 4. Save boxplot 
	plt.tight_layout()	
	plt.savefig(plot_file)
	plt.close()
