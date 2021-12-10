# !/usr/bin/env python
## created by Yun Hao @MooreLab 2021
## This script derives target binding profile of query compounds using the predictive features identified by feature selection pipeline 


## Module
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


## Main function 
def main(argv):
	## 0. Input arguments 
		# argv 1: input file that contains feature data of query compounds  
		# argv 2: input file that contains structure features predictive of target binding 
		# argv 3: name of data folder that contains training data for selection of predictive structure features
		# argv 4: name of output data folder  

	## 1. Read in query feature data file 
	data_df = pd.read_csv(argv[1], sep = '\t', header = 0, index_col = 0)

	## 2. Predict the target binding of query compounds using classifier built upon predictive features of each target
	# read in structure features predictive of target binding  
	feature_df = pd.read_csv(argv[2], sep = '\t', header = 0)		
	# iterate by row of predictive features
	data_pred_prob_dict = {}
	data_pred_label_dict = {}
	for i in range(0, feature_df.shape[0]):
		# obtain the target, measurement, and predictive feature information of current row  	
		i_target = feature_df.iloc[i,0].split('_')[0]
		i_measure = feature_df.iloc[i,0].split('_')[1]
		i_features = feature_df.iloc[i,1].split(',')
		# obtain name of file that contains training data for feature selection of the current target, read in training data 
		train_file = argv[3] + '_' + i_measure + '_0.25_binary_' + i_target + '_whole_data.tsv_train.tsv' 
		train_df = pd.read_csv(train_file, sep = '\t', header = 0, index_col = 0) 	
		# fit random forest classifier using the predictive structure features of training data 
		classifier = RandomForestClassifier(random_state = 0)
		classifier.fit(train_df[i_features].values, train_df[i_measure].values)
		# implement the classifier to predict target binding probability and label  
		data_pred_prob_dict[i_target] = classifier.predict_proba(data_df[i_features])[:,1]
		data_pred_label_dict[i_target] = classifier.predict(data_df[i_features])
	
	## 3. Build target binding profile of query compounds, output  
	# collect predicted binding probability profile of query compounds from all targets, output in data frame 
	data_pred_prob_df = pd.DataFrame(data_pred_prob_dict)
	data_pred_prob_df.index = data_df.index		
	data_pred_prob_df.to_csv(argv[4] + '_probability.tsv', sep = '\t', float_format = '%.5f')
	# collect predicted binding label profile of query compounds from all targets, output in data frame
	data_pred_label_df = pd.DataFrame(data_pred_label_dict)	
	data_pred_label_df.index = data_df.index
	data_pred_label_df.to_csv(argv[4] + '_label.tsv', sep = '\t')

	return 1


## Call main function
main(sys.argv) 
