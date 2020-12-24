# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script generates shell scripts that collects, analyzes, and visualizes feature selection results on compound target-adverse event datasets.


## functions
source("src/functions.R");

## 0. Input arguments
ss_file		<- "data/compound_structure_all_adverse_event_data/compound_target_descriptor_all_whole_data_summary.tsv"
ss_cut		<- 500;
input_folder	<- "data/compound_target_all_adverse_event_feature_select_implementation/descriptor_all/";
output_folder	<- "data/compound_target_all_adverse_event_feature_select_implementation/descriptor_all/";
optimal_file	<- "data/compound_target_all_adverse_event_feature_select_implementation/descriptor_all_all_adverse_event_optimal_hyperparameter_files.txt";
collect_name	<- "collect_selection_results_target_all_adverse_event";
analyze_name	<- "analyze_selection_results_target_all_adverse_event_tuning";

## 1. Obtain names of input files
#
ss_df <- read.delim(file = ss_file, sep = "\t", header = T);
ss_cut_id <- which(ss_df$N_samples > ss_cut);
all_aes <- unique(ss_df$Group[ss_cut_id]);
aa_len <- sapply(all_aes, function(aa) length(strsplit(aa, "_", fixed = T)[[1]]));
collect_commands <- mapply(function(aa, al){
	# 
	ip_file <- paste(input_folder, aa, "/", sep = "");
	op_file <- paste(input_folder, aa, sep = "")
	# 
	id1 <- 14 + al;
	id2 <- 13 + 2 * al;
	id3 <- 16 + 2 * al; 
	id4 <- 27 + 2 * al;
	command <- paste("Rscript", "src/collect_selection_results.R", ip_file, op_file, id1, id2, 1, id3, id4, sep = " ");
	return(command);
}, all_aes, aa_len);
# write shell scripts for jobs 
generate.parallel.bash.files(collect_commands, 1, collect_name, "src/run/");

## 2. 
setwd("~/project/TTox/");
analyze_commands <- sapply(all_aes, function(aa){
	# 
	aa_perf_file <- paste(output_folder, aa, "_performance_files.tsv", sep = "");
	aa_out_file <- paste(output_folder, aa, "_select_features", sep = "");
	command <- paste("python", "src/analyze_selection_results.py", aa_perf_file, "classification", "auc", "tuning", aa_out_file, "NA", sep = " ");
	return(command);
});
# write shell scripts for jobs
generate.parallel.bash.files(analyze_commands, 1, analyze_name, "src/run/");

## 3. 
setwd("~/project/TTox/");
opt_files <- sapply(all_aes, function(aa){
	aa_opt <- paste(output_folder, aa, "_select_features_optimal_hyperparameters.txt", sep = "");
	return(aa_opt);
});
writeLines(opt_files, optimal_file);
