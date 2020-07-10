# !usr/bin/Rscript 
# created by Yun Hao @MooreLab 2019
# This script collects results of feature selection.


## functions
source("src/functions.R");


## 0. Input arguments
Args		<- commandArgs(T);
input_folder	<- Args[1];     # input folder of performance files
output_folder	<- Args[2];     # output folder

## 1. Obtain names of performance files and data files 
# list all files in the folder
all_files <- list.files(input_folder);
# obtain all performance files 
perf_id <- sapply(all_files, function(af) length(strsplit(af, "performance", fixed = T)[[1]]));
all_perf_files <- all_files[perf_id == 2];
all_perf_files <- sapply(all_perf_files, function(apf) paste(input_folder, apf, sep = ""));
# obtain name of data files (target name + measurement type)
all_data_name <- sapply(all_perf_files, function(apf){
	apf_s <- strsplit(apf, "_")[[1]];
	# target name 
	target <- apf_s[[11]];
	# measurement type 
	measure <- apf_s[[10]];
	# combine 
	tm <- paste(target, measure, sep = "_");
	return(tm);
});
# obtain unique data files and corresponding performance files 
data_perf <- group.vector.by.categories(all_data_name, all_perf_files);

## 2. Obtain hyperparameter settings of performance files
# iterate by data file
data_hyper <- lapply(data_perf, function(dp){
	# iterate by performance file
	dp_hyper <- sapply(dp, function(d){
		d_s <- strsplit(d, "_")[[1]];
		# obtain hyperparamters 
		d_hyper <- paste(d_s[16:27], collapse = "_");
		return(d_hyper);
	});
	return(dp_hyper);
});
# obtain all hyperparamter settings 
all_hyper <- unique(unlist(data_hyper));
all_hyper <- sort(all_hyper);

## 3. Obtain performance metrics
# read in performance metric
perf_summary_list <- mapply(function(dp, dh){
	read.performance.files(dp, dh, all_hyper);
}, data_perf, data_hyper, SIMPLIFY = F);
# performance metrics of models dervied from all features 
all_perf_summary <- lapply(perf_summary_list, function(psl) psl[[1]]);
all_perf_summary_df <- data.frame(do.call(rbind, all_perf_summary));
write.table(all_perf_summary_df, paste(output_folder, "_all_features_summary.tsv", sep = ""), col.names = T, row.names = T, quote = F, sep = "\t");
# number of features of model derived from feature selection 
select_number_summary <- lapply(perf_summary_list, function(psl) psl[[2]]);
select_number_summary_df <- data.frame(do.call(rbind, select_number_summary));
write.table(select_number_summary_df, paste(output_folder, "_select_features_number_summary.tsv", sep = ""), col.names = T, row.names = T, quote = F, sep = "\t");
# cross-validation performance on training data of model derived from feature selection 
select_train_perf_summary <- lapply(perf_summary_list, function(psl) psl[[3]]);
select_train_perf_summary_df <- data.frame(do.call(rbind, select_train_perf_summary));
write.table(select_train_perf_summary_df, paste(output_folder, "_select_features_training_performance_summary.tsv", sep = ""), col.names = T, row.names = T, quote = F, sep = "\t");
# performance on training data of model derived from feature selection 
select_test_perf_summary <- lapply(perf_summary_list, function(psl) psl[[4]]);
select_test_perf_summary_df <- data.frame(do.call(rbind, select_test_perf_summary));
write.table(select_test_perf_summary_df, paste(output_folder, "_select_features_testing_performance_summary.tsv", sep = ""), col.names = T, row.names = T, quote = F, sep = "\t");
