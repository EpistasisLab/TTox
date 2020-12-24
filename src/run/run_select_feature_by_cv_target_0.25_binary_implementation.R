# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script generates shell scripts that implement feature selection pipeline on compound structure-target binding datasets.


## functions
source("src/functions.R");

## 0. Input arguments
Args			<- commandArgs(T);
descriptor_type		<- Args[1];     # input folder of all descriptor files
hyperparameter_file	<- Args[2];	# input file of optimal hyperparameter setting 
N_cores			<- Args[3];	# number of cores to run the jobs 
# data folders
target_folder		<- "data/compound_target_0.25_binary_data/";
chemical_folder		<- "https://raw.githubusercontent.com/yhao-compbio/chemical/master/data/offsides_compounds/";
output_folder		<- "data/compound_target_0.25_binary_feature_select_implementation/";
job_name		<- "select_feature_by_cv_target_0.25_binary_implementation";

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

## 3. Read in optimal hyperparamter settings 
hp_lines <- readLines(hyperparameter_file);
hp_list <- sapply(hp_lines, function(hl) strsplit(hl, ": ", fixed = T)[[1]][[2]]);
hps <- paste(hp_list, collapse = " ");

## 3. Generate commands for jobs 
commands <- mapply(function(af, afi1, afi2){
	# training file
	af_train <- paste(target_folder, af, "_train.tsv", sep = "");
	# testing file
	af_test <- paste(target_folder, af, "_test.tsv", sep = "");
	# prediction file 
	af_pred <- paste(chemical_folder, strsplit(afi1, "_")[[1]][[1]], "_combined/offsides_compounds_", afi1, ".tsv", sep = "")	
	# output prefix 
	af_out <- paste(output_folder, afi1, "/", af, sep = "")
	af_commands <- paste("python", "src/select_feature_by_cv.py", af_train, af_test, af_pred, af_out, afi2, hps, sep = " ")
	return(af_commands);
}, all_files, all_files_info[1,], all_files_info[2,])
# shuffle commands (in order to balance running time of each shell scripts)
ran_id <- sample(1:length(commands), length(commands));
commands <- commands[ran_id];
# write shell scripts for jobs
generate.parallel.bash.files(commands, as.integer(N_cores), paste(job_name, descriptor_type, sep = "_"), "src/run/");
