# !/usr/bin/env python
## created by Yun Hao @MooreLab 2019
## This script contains functions for computing structure similarity between compounds.  

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
