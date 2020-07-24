# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains functions for computing structure similarity between compounds, and feature similarity between models.  


## Module
import numpy as np
import pandas as pd


## This function computes Tanimoto Coefficient between two sets of chemical fingerprints 
def compute_tanimoto_coefficient(vec1, vec2):
	## 0. Input arguments: 
		# vec1: numpy array that contains chemical fingerprint of compound 1
		# vec2: numpy array that contains chemical fingerprint of compound 2

	## 1. Compute Tanimoto Coefficient (#overlap/#union)
	vec_sum = vec1 + vec2
	N_overlap = np.sum(vec_sum == 2)
	N_union = np.sum(vec_sum == 1) + N_overlap
	if N_union == 0:
		tc = 0
	else:
		tc = N_overlap/N_union
	
	return tc
	

## This function computes pairwise Tanimoto Coefficient between compounds within a give set 
def compute_tanimoto_coefficients_within_set(set_df, set_name):
	## 0. Input arguments: 
		# set_df: data frame that contains chemical fingerprints of compounds 
		# set_name: name of the compound set 
	
	## 1. Compute pairwise Tanimoto Coefficient between compounds in the set 
	# obtain names of compounds in the set
	all_compounds = set_df.index
	# iterate by compound 
	compound_max_sim = []
	compound_mean_sim = []
	for ac in all_compounds: 
		# obtain fingerprints of the compound
		ac_values = set_df.loc[ac,:].values
		# obtain fingerprints of other compounds
		others_df = set_df.drop(ac, axis = 0)
		# otbain names of other compounds
		other_compounds = others_df.index
		# iterate by other compound 
		oc_sim = []
		for oc in other_compounds:
	 		# compute Tanimoto Coefficient between the two compounds
			oc_tc = compute_tanimoto_coefficient(ac_values, others_df.loc[oc,:].values)
			oc_sim.append(oc_tc)
		# take the maximum across all other compounds 
		compound_max_sim.append(np.max(oc_sim))
		# take the average across all other compounds 
		compound_mean_sim.append(np.mean(oc_sim))	
	
	## 2. Output in data frame (Four columns, 1: name of compound set, 2: name of compounds, 3: max similarity, 4: average similarity)
	out_df = pd.DataFrame({'comparing_sets': set_name, 'compound_name': all_compounds, 'max_tc': compound_max_sim, 'mean_tc': compound_mean_sim})

	return out_df


##  This function computes pairwise Tanimoto Coefficient between compounds from two different sets 
def compute_tanimoto_coefficients_between_sets(set1_df, set2_df, sets_name):
	## 0. Input arguments:
		# set1_df: data frame that contains chemical fingerprints of compounds in set 1 (set of interest)
		# set2_df: data frame that contains chemical fingerprints of compounds in set 2 (set to be compared with)
		# sets_name: name of the two compound sets
	
	## 1. Compute pairwise Tanimoto Coefficient between compounds from two different sets
	# obtain names of compounds in two set s
	set1_compounds = set1_df.index
	set2_compounds = set2_df.index
	# iterate by compound in set 1 
	compound_max_sim = []
	compound_mean_sim = []
	for s1c in set1_compounds:
		# obtain fingerprints of the compound in set 1
		s1c_values = set1_df.loc[s1c,:].values
		# iterate by compond in set 2
		s2c_sim = []
		for s2c in set2_compounds:
			# compute Tanimoto Coefficient between the two compounds
			s2c_tc = compute_tanimoto_coefficient(s1c_values, set2_df.loc[s2c,:].values)
			s2c_sim.append(s2c_tc)
		# take the maximum across all other compounds 
		compound_max_sim.append(np.max(s2c_sim))
		# take the average across all other compounds 
		compound_mean_sim.append(np.mean(s2c_sim))

	## 2. Output in data frame
	out_df = pd.DataFrame({'comparing_sets': sets_name, 'compound_name': set1_compounds, 'max_tc': compound_max_sim, 'mean_tc': compound_mean_sim})

	return out_df


## This function computes pairwise similarity of selected features between a set of models  
def compute_model_feature_similarity(select_feature_df, index_column, select_column):
	## 0. Input arguments: 
		# select_feature_df: data frame that contains the index and selected features of each model 
		# index_column: name of column that contains the index of each model 
		# select_column: name of column that contains the selected features of each model 
		 
	## 1. Obtain number of models
	N_models = select_feature_df.shape[0]	

	## 2. Compute pairwise feature similarity among models
	# iterate by model pair  
	index1 = []
	index2 = []
	pw_sim_vec = []
	for i in range(0, N_models-1):
		# obtain index of model 1
		index_i = select_feature_df[index_column].iloc[i, ]
		# obtain selected relevant features of model 1 
		feat_i = select_feature_df[select_column].iloc[i, ]
		feat_i_set = set(feat_i.split(','))
		for j in range(i+1, N_models):
			# obtain index of model 2
			index_j = select_feature_df[index_column].iloc[j, ]
			# obtain selected relevant features of model 2
			feat_j = select_feature_df[select_column].iloc[j, ]
			feat_j_set = set(feat_j.split(','))
			# compute jaccard similarity of two features sets (#intersect/#union) 
			N_inter = len(feat_i_set.intersection(feat_j_set))
			N_union = len(feat_i_set.union(feat_j_set))
			jaccard_index_ij = N_inter/N_union 
			index1.append(index_i)
			index2.append(index_j)
			pw_sim_vec.append(jaccard_index_ij)
	# output in data frame format
	sim_df = pd.DataFrame({'index1': index1, 'index2':index2, 'feature_jaccard_similarity':pw_sim_vec})
	
	return sim_df


## This function groups pairwise similarity scores by specified classes of models
def group_pairwise_similarity_by_class(pw_sim, index_class):
	## 0. Input arguments:
		# pw_sim: array that contains pairwise similarity scores among models 
		# index_class: array that contains class of models 

	## 1. Obtain set of unique classes
	N_index = len(index_class)
	index_class_set = set(index_class)	

	## 2. Group pairwise similarity by class 
	class_sim_dict = {}
	# iterate by unique class 
	for ics in index_class_set:
		# find all indices of models that belong to the class  
		ics_id = [index for index, value in enumerate(index_class) if value == ics]	
		N_ics = len(ics_id)
		if N_ics > 1:
			# find indices of model pairs that both belong to the class 
			class_sim_dict[ics] = []
			for ii in range(0, N_ics-1):
				ii_id = ics_id[ii]
				for ij in range(ii+1, N_ics):
					ij_id = ics_id[ij]		
					pair_id = int((2 * N_index - 1 - ii_id) * ii_id/2 + ij_id - ii_id - 1)
					# Add similarity scores of the identified pairs to the output dictionary 
					class_sim_dict[ics].append(pw_sim[pair_id])

	return class_sim_dict
