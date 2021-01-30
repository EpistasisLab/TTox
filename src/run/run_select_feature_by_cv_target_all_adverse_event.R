# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script generates shell scripts that implement feature selection pipeline on all compound target-adverse event datasets.


## functions
source("src/functions.R");

## 0. Input arguments
ss_file		<- "data/compound_structure_all_adverse_event_data/compound_target_descriptor_all_whole_data_summary.tsv"
ss_cut		<- 500;
input_folder	<- "data/compound_target_all_adverse_event_data/";
output_folder	<- "data/compound_target_all_adverse_event_feature_select_implementation/";
pred_folder	<- "data/compound_target_0.25_binary_feature_select_implementation/";
job_name	<- "select_feature_by_cv_target_all_adverse_event_implementation";
N_cores		<- 150;

## 1. Obtain names of input target-adverse event files
# read in data frame that contains sample size information adverse event data
ss_df <- read.delim(file = ss_file, sep = "\t", header = T);
# select adverse event data that pass minimum sample size requirement 
ss_cut_id <- which(ss_df$N_samples > ss_cut); 
all_aes <- unique(ss_df$Group[ss_cut_id]);
# AUC thresholds of target binding profile 
all_mcs <- c("0.85");
# combine requirements of target and adverse event to obtain input target-adverse event files
all_ae_vec <- rep(all_aes, each = length(all_mcs));
all_mc_vec <- rep(all_mcs, times = length(all_aes));
all_files <- mapply(function(aav, amv){
	paste("compound_target_descriptor_all_mc", amv, aav, "whole_data.tsv", sep = "_");
}, all_ae_vec, all_mc_vec);

## 2. Make folders to store output files
# make one folder for each feature type   
feat_types <- "descriptor_all";
ft_name <- paste(output_folder, feat_types, "/", sep = "");
system(paste("mkdir", ft_name, sep = " "));
# make one sub folder for each adverse event 
ftf_aes <- sapply(all_aes, function(aa){
	aa_name <- paste(ft_name, aa, "/", sep = "");
	system(paste("mkdir", aa_name, sep = " "));
	return(1);
});

## 3. Generate different parts of commands  
part <- NULL
# generate parts that include training file, testing file, prediction file, output prefix, and label column 
part[[1]] <- mapply(function(af, amv, aav){
	# obtain name of training file
	af_train <- paste(input_folder, af, "_train.tsv", sep = "");
	# obtain name of testing file
	af_test <- paste(input_folder, af, "_test.tsv", sep = "");
	# obtain name of prediction file 
	af_pred <- paste(pred_folder, feat_types, "_analysis/", feat_types, "_select_features_mc_", amv, "_offsides_compounds_binding_affinity_prediction_select_features.tsv", sep = "")	
	# obtain name of output prefix 
	af_out <- paste(output_folder, feat_types, "/", aav, "/", af, "_mc_", amv, sep = "")
	af_commands <- paste("python", "src/select_feature_by_cv.py", af_train, af_test, af_pred, af_out, 'toxicity_label', sep = " ")
	return(af_commands);
}, all_files, all_mc_vec, all_ae_vec);
# generate parts that include number of folds and ranking methods, whether to include TURF, remove percentage, and supervised learning task  
part[[2]] <- c("10 MultiSURF 0 0.1 classification");
# generate parts that include supervised learning methods, tolerance , consistency score threshold, number of repeat runs, and whether to predict probability of positive class 
part[[3]] <- c("RandomForest 50 0.5 20 1");

## 4. Combine different parts to generate commands for jobs 
commands <- generate.all.possible.hyperparameter.combinations(part);
# shuffle commands (in order to balance running time of each shell scripts)
ran_id <- sample(1:length(commands), length(commands));
commands <- commands[ran_id];
# write shell scripts for jobs
generate.parallel.bash.files(commands, as.integer(N_cores), job_name, "src/run/");
