# !/usr/bin/env Rscript
# created by Yun Hao @MooreLab 2019
# This script analyzes similarity of selected relevant features among different models. 
 

## Module 
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, 'src/')
import ttox_sim
import ttox_plot


# Main function 
def main(argv):
	## 0. Input arguments:
		# argv 1: input file that contains selected relevant features of all models (row: models, column 1: model index, column 2: features) 
		# argv 2: output file of analysis results 
		# argv 3: output file of visualization results
		# argv 4: input feature type: 'structure' or 'target'
			
	## 1. Read in selected features of models
	feature_df = pd.read_csv(argv[1], sep = '\t', header = 0)

	## 2. Perform similarity analysis according to the input feature type  
	# analyze structure feature similarity 
	if argv[4] == 'structure':
		# obtain the target and measurement name of each model 
		measure_class = []
		target = []
		for fdtmv in feature_df.target_measurement.values:
			target.append(fdtmv.split('_')[0])
			measure_class.append(fdtmv.split('_')[1])
		# build data frame that contains target-structure feature relationships 
		target_feature_df = pd.DataFrame({'target': target, 'structure_features': feature_df.structure_features.values})	
		# read in curated classes of proteins in human druggable genome 
		protein_class = pd.read_csv('https://raw.githubusercontent.com/yhao-compbio/target/master/data/human_druggable_genome_protein_class.tsv', sep = '\t', header = 0)
		# merge data frames to obtain the curated classes of model targets
		target_class_feature_df = pd.merge(target_feature_df, protein_class, left_on = 'target', right_on = 'uniprot_id')
		# compute pairwise similarity of selected relevant features among targets 
		feature_sim_df = ttox_sim.compute_model_feature_similarity(feature_df, 'target_measurement', 'structure_features')
		feature_sim_df.to_csv(argv[2] + '_feature_similarity.tsv', sep = '\t', index = False)
		# group pairwise similarity scores by measurement class
		measure_sim_dict = ttox_sim.group_pairwise_similarity_by_class(feature_sim_df.feature_jaccard_similarity.values, measure_class)			 
		# compare and visualize the distribution of pairwise similarity scores across different measurement classes 
		measure_compare = ttox_plot.visualize_similarity_density_comparison(measure_sim_dict, feature_sim_df.feature_jaccard_similarity.values, argv[3] + '_measure_similarity')	
		# group pairwise similarity scores by functional class of target
		feature_sim_df1 = ttox_sim.compute_model_feature_similarity(target_class_feature_df, 'target', 'structure_features')
		class_sim_dict = ttox_sim.group_pairwise_similarity_by_class(feature_sim_df1.feature_jaccard_similarity.values, target_class_feature_df['class'].values)
		# compare and visualize the distribution of pairwise similarity scores across different functional classes
		class_compare = ttox_plot.visualize_similarity_density_comparison(class_sim_dict, feature_sim_df.feature_jaccard_similarity.values, argv[3] + '_function_similarity')  
		
	return 1


## Call main function
main(sys.argv)
