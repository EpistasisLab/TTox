# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script generates shell scripts that run feature selection on compound-target interaction datasets.


## functions
source("src/functions.R");


## 1. Obtain names of input files
# list all files in the folder
all_files <- list.files('data/compound_target_data/');
# select whole dataset files 
whole_id <- sapply(all_files, function(af) length(strsplit(af, "whole_data.tsv", fixed = T)[[1]]));
all_files <- all_files[whole_id == 1];
# exclude summary files 
sum_id <- sapply(all_files, function(af) length(strsplit(af, "summary", fixed = T)[[1]]));
all_files <- all_files[sum_id == 1];

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
	afi1_s <- strsplit(afi1, "_")[[1]][[1]];
	af_pred <- paste("/home/yunhao1/project/chemical/data/offsides_compounds/", afi1_s, "_combined/offsides_compounds_", afi1, ".tsv", sep = "")	
	# output prefix 
	af_out <- paste("data/compound_target_feature_select/", afi1, "/", af, sep = "")
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
# parrts that include consistency score threshold 
part[[6]] <- c(0.5, 0.7)

## 4. Genrate all commands for jobs 

## 2. Generate commands for jobs 
commands <- generate.all.possible.hyperparameter.combinations(part);
# shuffle commands (in order to balance running time of each shell scripts)
set.seed(0);
ran_id <- sample(1:length(commands), length(commands));
commands <- commands[ran_id];
# write shell scripts for jobs
generate.parallel.bash.files(commands, 30, 'select_feature_by_cv_target', "src/run/");
