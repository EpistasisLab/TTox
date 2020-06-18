# !/usr/bin/env Rscript
# created by Yun Hao @MooreLab 2019
# This script computes and visualizes structure similarity between compounds. 
 

## Module 
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, 'src/')
import ttox_sim

 
# Main function 
def main(argv):
	## 0. Input arguments:
		# argv 1: name of training file 
		# argv 2: name of testing file
		# argv 3: name of label(resonse) column
		# argv 4: name of output file 
		# argv 5: name of plot file 
		
	## 1. Read in learning data
	# read in training data
	train_df = pd.read_csv(argv[1], sep = '\t', header = 0, index_col = 0)
	# read in testing data
	test_df = pd.read_csv(argv[2], sep = '\t', header = 0, index_col = 0)
	# combine training and testing data 
	all_df = pd.concat([train_df, test_df]) 
	
	## 2. Compute similarity within each set 
	# within all compunds 
	all_vs_all = ttox_sim.compute_tanimoto_coefficients_within_set(all_df.drop(argv[3], axis = 1), 'all vs all')
	# within training compounds
	train_vs_train = ttox_sim.compute_tanimoto_coefficients_within_set(train_df.drop(argv[3], axis = 1), 'train vs train')
	# within testing compounds
	test_vs_test = ttox_sim.compute_tanimoto_coefficients_within_set(test_df.drop(argv[3], axis = 1), 'test vs test')

	## 3. Compute similarity between training and testing set
	test_vs_train = ttox_sim.compute_tanimoto_coefficients_between_sets(test_df.drop(argv[3], axis = 1), train_df.drop(argv[3], axis = 1), 'test vs train')
	
	## 4. Combine results and output 
	# combine within-set and between-set results
	all_sim_results = pd.concat([all_vs_all, train_vs_train, test_vs_test, test_vs_train])	
	# output 
	output_result_file = argv[4] + '_similarity_results.tsv' 	
	all_sim_results.to_csv(output_result_file, sep = '\t', index = False, float_format = '%.5f')	

	## 5. Visualize similarity comparison by boxplots
	# set figure size and font size  
	plt.figure(figsize = (15,20))
	plt.rc('font', size = 10)
	plt.rc('axes', titlesize = 34)
	plt.rc('axes', labelsize = 34)
	plt.rc('xtick', labelsize = 34)
	plt.rc('ytick', labelsize = 34)
	plt.rc('legend', fontsize = 24)
	# plot comprison of maximum similairty 
	plt.subplot(211)
	sns.boxplot(x = 'comparing_sets', y = 'max_tc', data = all_sim_results, width = 0.5)
	plt.ylim(-0.05, 1.05)
	plt.xlabel('Sets')
	plt.ylabel('Maximum similarity')
	# plot comprison of average similairty 
	plt.subplot(212)
	sns.boxplot(x = 'comparing_sets', y = 'mean_tc', data = all_sim_results, width = 0.5)
	plt.ylim(-0.05, 1.05)
	plt.xlabel('Sets')
	plt.ylabel('Average similarity')
	# output plot to pdf file
	plt.tight_layout()
	plt.savefig(argv[5] + '_similarity_boxplot.pdf')
	plt.close()

	return 1


## Call main function
main(sys.argv)
