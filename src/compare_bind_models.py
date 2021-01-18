# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script compares and visualizes target-binding prediction models across distinct groups
 

## Module 
import sys
import numpy as np
import pandas as pd 
sys.path.insert(0, 'src/')
import ttox_sim
import ttox_plot


# Main function 
def main(argv):
	## 0. Input arguments:
		# argv 1: input file that contains selected relevant features of all models (row: models, column 1: model index, column 2: features) 
		# argv 2: input file that contains testing performance of all models (row: models)
		# argv 3: output file of analysis results 
		# argv 4: output file of visualization results
			
	## 1. Compare and visualize testing performance across distinct model groups (by measurement type, by target class) 
	# read in testing performance of target binding prediction models  
	testing_perf_df = pd.read_csv(argv[2], sep = '\t', header = 0, index_col = 0)
	perf_col = testing_perf_df.columns[0]
	# obtain the target and measurement name of each model 
	target = []
	measure_class = []
	for tpdi in testing_perf_df.index:
		target.append(tpdi.split('_')[0])
		measure_class.append(tpdi.split('_')[1])
	# build data frame that contains target-structure feature relationships 
	target_measure_df = pd.DataFrame({'target': target, 'measure': measure_class, 'target_measurement': testing_perf_df.index, perf_col: testing_perf_df[perf_col].values})
	# compare and visualize testing performance across different measurements 
	measure_od = ['pKi', 'pIC50', 'pKd', 'pEC50'] 
	measure_compare = ttox_plot.visualize_group_performance_comparison(target_measure_df, perf_col, 'measure', 'measurement', measure_od, argv[4] + '_performance')
	# read in curated classes of target proteins in human druggable genome 
	protein_class = pd.read_csv('https://raw.githubusercontent.com/yhao-compbio/target/master/data/human_druggable_genome_protein_class.tsv', sep = '\t', header = 0)
	protein_class = protein_class[protein_class['class'] != 'Other protein']
	# merge class and model info of targets 
	target_measure_class_df = pd.merge(target_measure_df, protein_class, how = 'left', left_on = 'target', right_on = 'uniprot_id')
	target_measure_class_df = target_measure_class_df.fillna('Others')
	# compare and visualize testing performance across different target classes 
	class_od = ['Catalytic receptor', 'G protein coupled receptor', 'Transporter', 'Ion channel', 'Enzyme', 'Nuclear hormone receptor', 'Others']
	class_compare = ttox_plot.visualize_group_performance_comparison(target_measure_class_df, perf_col, 'class', 'Functional class', class_od, argv[4] + '_performance')
	
	## 2. Perform similarity analysis according to the input feature type  	
	# read in selected features of target binding prediction models  
	feature_df = pd.read_csv(argv[1], sep = '\t', header = 0)
	# compute pairwise similarity of selected features among all models 
	feature_sim_df = ttox_sim.compute_model_feature_similarity(feature_df, 'target_measurement', 'structure_features')
	feature_sim_df.to_csv(argv[3] + '_feature_similarity.tsv', sep = '\t', index = False)

	## 3. Visualize class composition of all model targets   
	# merge data frames to obtain the curated classes of model targets
	target_class_feature_df = pd.merge(target_measure_class_df, feature_df, on = 'target_measurement')
	# visualize class composition by pie chart 
	class_prop = ttox_plot.visualize_group_proportion(target_class_feature_df, 'class', 'Functional class of targets', class_od, argv[4] + '_target_proportion')

	## 4. Compare and visualize intergroup/intragroup similarity of selected features among funciton classes
	# group pairwise similarity scores by functional class of target
	target_class_feature_df = target_class_feature_df[target_class_feature_df['class'] != 'Others']
	feature_sim_df1 = ttox_sim.compute_model_feature_similarity(target_class_feature_df, 'target', 'structure_features')
	class_sim_df, class_sim_pv = ttox_sim.group_pairwise_similarity_by_class(feature_sim_df1.feature_jaccard_similarity.values, target_class_feature_df['class'].values)
	# compare and visualize intergroup/intragroup similarity of selected features among function classes
	class_od = ['Catalytic receptor', 'G protein coupled receptor', 'Transporter', 'Ion channel', 'Enzyme', 'Nuclear hormone receptor']
	class_feature_compare = ttox_plot.visualize_group_feature_similarity_comparison(class_sim_df, class_sim_pv, 'class', 'Functional class', class_od, argv[4] + '_function_similarity')  
		
	return 1


## Call main function
main(sys.argv)
