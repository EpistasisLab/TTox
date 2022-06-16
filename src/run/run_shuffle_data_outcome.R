# !/usr/bin/env Rscript
## created by Yun Hao @MooreLab 2021
## This script generates shell scripts that shuffle outcomes labels of input feature-response data 


## functions
source("src/functions.R");


## 0. Input arguments
Args			<- commandArgs(T);
input_data_folder	<- Args[1];		# folder name that contains input feature-response data files 
output_folder		<- Args[2];		# folder name of output data file 
job_name		<- Args[3];		# job name
outcome_col		<- "assay_outcome";	# name of label (response) column
N_shuffle		<- 200;			# number of shuffled data files to be generated 

## 1. Obtain names of input data files
# list all files in the data folder
all_data_files <- list.files(input_data_folder);
# select whole dataset files 
whole_id <- sapply(all_data_files, function(adf) length(strsplit(adf, "whole_data.tsv", fixed = T)[[1]]));
all_data_files <- all_data_files[whole_id == 1];
# exclude summary files 
sum_id <- sapply(all_data_files, function(adf) length(strsplit(adf, "summary", fixed = T)[[1]]));
all_data_files <- all_data_files[sum_id == 1];

## 2. Generate commands for jobs 
commands <- mapply(function(adf){
	# input training feature-response data file
	adf_train <- paste(input_data_folder, adf, "_train.tsv", sep = "");
	# output shuffled data file 
	adf_output <- paste(output_folder, adf, "_train.tsv", sep = "");
	# generate command 
	adf_command <- paste("python", "src/shuffle_data_outcome.py", adf_train, outcome_col, N_shuffle, adf_output, sep = " ");
	return(adf_command);
}, all_data_files);
# write shell scripts for jobs
generate.parallel.bash.files(commands, 1, job_name, "src/run/");
