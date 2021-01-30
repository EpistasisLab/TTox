# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script generates shell scripts that implement L1 regularization on compound structure-target binding datasets.


## functions
source("src/functions.R");

## 0. Input arguments
Args		<- commandArgs(T);
descriptor_type	<- Args[1];     # input folder of all descriptor files
N_cores		<- Args[2];	# number of cores to run the jobs 
# data_folders
target_folder	<- "data/compound_target_0.25_binary_data/";
output_folder	<- "data/compound_target_0.25_binary_regularization_implementation/";
job_name	<- "select_feature_by_regularization_target_0.25_binary_implementation";

## 1. Obtain names of input files
# list all files in the folder
all_files <- list.files(target_folder);
# select whole dataset files 
whole_id <- sapply(all_files, function(af) length(strsplit(af, "whole_data.tsv", fixed = T)[[1]]));
all_files <- all_files[whole_id == 1];
# exclude summary files 
sum_id <- sapply(all_files, function(af) length(strsplit(af, "summary", fixed = T)[[1]]));
all_files <- all_files[sum_id == 1];
# obtain files of the descriptor type
descriptor_id <- sapply(all_files, function(af) length(strsplit(af, descriptor_type, fixed = T)[[1]]));
all_files <- all_files[descriptor_id == 2];

## 2. Obtain file information 
all_files_info <- mapply(function(af){
	af_s <- strsplit(af, "_", fixed = T)[[1]];
	# descriptor type
	dt <- paste(af_s[3:4], collapse = "_");
	# measurement type 
	mt <- af_s[[5]];
	return(c(dt, mt));
}, all_files, SIMPLIFY = T);

## 3. Specify hyperparameters
hps <- '10 classification 20';

## 3. Generate commands for jobs 
commands <- mapply(function(af, afi1, afi2){
	# training file
	af_train <- paste(target_folder, af, "_train.tsv", sep = "");
	# testing file
	af_test <- paste(target_folder, af, "_test.tsv", sep = "");
	# prediction file 
	af_pred <- NA	
	# output prefix 
	af_out1 <- paste(output_folder, afi1, "/lasso/", af, sep = "")
	af_out2 <- paste(output_folder, afi1, "/randomforest/", af, sep = "")
	af_commands <- paste("python", "src/select_feature_by_regularization.py", af_train, af_test, af_pred, af_out1, af_out2, afi2, hps, sep = " ")
	return(af_commands);
}, all_files, all_files_info[1,], all_files_info[2,])
# shuffle commands (in order to balance running time of each shell scripts)
ran_id <- sample(1:length(commands), length(commands));
commands <- commands[ran_id];
# write shell scripts for jobs
generate.parallel.bash.files(commands, as.integer(N_cores), paste(job_name, descriptor_type, sep = "_"), "src/run/");
