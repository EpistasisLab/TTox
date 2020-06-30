# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script generates shell scripts that run feature selection hyperparamter tuning on compound-target interaction datasets.


## functions
source("src/functions.R");

## 0. Input arguments
Args		<- commandArgs(T);
descriptor_type	<- Args[1];     # input folder of all descriptor files
N_files		<- Args[2];	# number of files to choose for hyperparameter tuning
N_cores		<- Args[3];	# number of cores to run the jobs 

## 1. Obtain names of input files
# list all files in the folder
all_files <- list.files('data/compound_target_data/');
# select whole dataset files 
whole_id <- sapply(all_files, function(af) length(strsplit(af, "whole_data.tsv", fixed = T)[[1]]));
all_files <- all_files[whole_id == 1];
# exclude summary files 
sum_id <- sapply(all_files, function(af) length(strsplit(af, "summary", fixed = T)[[1]]));
all_files <- all_files[sum_id == 1];
# obtain files of the descriptor type
descriptor_id <- sapply(all_files, function(af) length(strsplit(af, descriptor_type, fixed = T)[[1]]));
all_files <- all_files[descriptor_id == 2];
# choose files for parameter tuning
set.seed(0);
subset_id <- sample(1:length(all_files), as.integer(N_files));
all_files <- all_files[subset_id];
# output file name
all_files_wd <- sapply(all_files, function(af) paste("data/compound_target_data/", af, "_train.tsv", sep = ""))
writeLines(all_files_wd, paste("data/compound_target_feature_select_tuning/", descriptor_type, "_sample_files.txt", sep = "")) 

## 2. Obtain file information 
all_files_info <- mapply(function(af){
	af_s <- strsplit(af, "_", fixed = T)[[1]];
	# descriptor type
	dt <- paste(af_s[3:4], collapse = "_");
	# measurement type 
	mt <- af_s[[5]];
	return(c(dt, mt));
}, all_files, SIMPLIFY = T);

## 3. Generate parts of commands  
part <- NULL
# parts that include training file, testing file, prediction file, output prefix, label column, and numer of folds 
part[[1]] <- mapply(function(af, afi1, afi2){
	# training file
	af_train <- paste("data/compound_target_data/", af, "_train.tsv", sep = "");
	# testing file
	af_test <- paste("data/compound_target_data/", af, "_test.tsv", sep = "");
	# prediction file 
	af_pred <- "NA"	
	# output prefix 
	af_out <- paste("data/compound_target_feature_select_tuning/", afi1, "/", af, sep = "")
	af_commands <- paste("python", "src/select_feature_by_cv.py", af_train, af_test, af_pred, af_out, afi2, 10, sep = " ")
	return(af_commands);
}, all_files, all_files_info[1,], all_files_info[2,])
# parts that include ranking methods 
part[[2]] <- c("MultiSURF", "MultiSURFstar");
# parts that inclue whether to include TURF, remove percentage, supervised learning task 
part[[3]] <- c("0 0.1 regression", "1 0.1 regression");
# parts that include supervised learning methods
part[[4]] <- c("RandomForest", "XGBoost");
# parts that include tolerance 
part[[5]] <- c(20, 50);
# parrts that include consistency score threshold, number of repeat runs 
part[[6]] <- c('0.5 20', '0.7 20')

## 4. Generate commands for jobs 
commands <- generate.all.possible.hyperparameter.combinations(part);
# shuffle commands (in order to balance running time of each shell scripts)
ran_id <- sample(1:length(commands), length(commands));
commands <- commands[ran_id];
# write shell scripts for jobs
generate.parallel.bash.files(commands, as.integer(N_cores), paste('select_feature_by_cv_target_tuning', descriptor_type, sep = "_"), "src/run/");
