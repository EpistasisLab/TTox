# !usr/bin/Rscript 
# created by Yun Hao @MooreLab 2019
# This script collects feature selection results of multiple datasets.


## functions
source("src/functions.R");

## 0. Input arguments
Args		<- commandArgs(T);
input_folder	<- Args[1];     	# input folder of performance files
output_folder	<- Args[2];     	# output folder
name_lower_id	<- as.integer(Args[3]);	# lower bound index of data file name 	
name_upper_id	<- as.integer(Args[4]);	# upper bound index of data file name 
consecutive	<- as.integer(Args[5]); # whether lower bound and upper bound of data file name are consecutive (0: no or 1: yes)
hyper_lower_id	<- as.integer(Args[6]);	# lower bound index of hyperparameter in the file name 
hyper_upper_id	<- as.integer(Args[7]);	# upper bound index of hyperparameter in the file name 

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
	# extract name   
	if(consecutive == 1)	tm <- paste(apf_s[name_lower_id:name_upper_id], collapse = "_")
	else	tm <- paste(apf_s[[name_lower_id]], apf_s[[name_upper_id]], sep = "_")
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
		d_hyper <- paste(d_s[hyper_lower_id:hyper_upper_id], collapse = "_");
		return(d_hyper);
	});
	return(dp_hyper);
});
# obtain all hyperparamter settings 
all_hyper <- unique(unlist(data_hyper));
all_hyper <- sort(all_hyper);

## 3. Obtain performance metrics
# read in performance metric
perf_read_list <- mapply(function(dp, dh){
	read.performance.files(dp, dh, all_hyper);
}, data_perf, data_hyper, SIMPLIFY = F);

# number of features of model derived from feature selection 
select_number_summary <- lapply(perf_read_list, function(prl) prl[[1]]);
select_number_summary_df <- data.frame(do.call(rbind, select_number_summary));
write.table(select_number_summary_df, paste(output_folder, "_select_features_number_summary.tsv", sep = ""), col.names = T, row.names = T, quote = F, sep = "\t");
# obtain performance metrics 
all_metrics <- perf_read_list[[1]][[2]];
# iterate by metrics 
all_metric_perf <- mapply(function(am, lam){ 
	# performance on testing data of models derived from all features 
	all_perf_summary <- lapply(perf_read_list, function(prl) prl[[3]][[lam]][[1]]);
	all_perf_summary_df <- data.frame(do.call(rbind, all_perf_summary));
	all_perf_summary_file <- paste(output_folder, "_all_features_summary_", am, ".tsv", sep = ""); 
	write.table(all_perf_summary_df, all_perf_summary_file, col.names = T, row.names = T, quote = F, sep = "\t");
	# cross-validation performance on training data of model derived from feature selection 
	select_train_perf_summary <- lapply(perf_read_list, function(prl) prl[[3]][[lam]][[2]]);
	select_train_perf_summary_df <- data.frame(do.call(rbind, select_train_perf_summary));
	select_train_perf_summary_file <- paste(output_folder, "_select_features_training_performance_summary_", am, ".tsv", sep = "");
	write.table(select_train_perf_summary_df, select_train_perf_summary_file, col.names = T, row.names = T, quote = F, sep = "\t");
	# performance on training data of model derived from feature selection 
	select_test_perf_summary <- lapply(perf_read_list, function(prl) prl[[3]][[lam]][[3]]);
	select_test_perf_summary_df <- data.frame(do.call(rbind, select_test_perf_summary));
	select_test_perf_summary_file <- paste(output_folder, "_select_features_testing_performance_summary_", am, ".tsv", sep = "");
	write.table(select_test_perf_summary_df, select_test_perf_summary_file, col.names = T, row.names = T, quote = F, sep = "\t");
	# output file names  
	out_files <- c(all_perf_summary_file, select_train_perf_summary_file, select_test_perf_summary_file);
	names(out_files) <- c("all_test", "select_train", "select_test");
	return(out_files)
}, all_metrics, 1:length(all_metrics));
colnames(all_metric_perf) <- all_metrics;
write.table(all_metric_perf, file = paste(output_folder, "_performance_files.tsv", sep = ""), col.names = T, row.names = T, quote = F, sep = "\t");
