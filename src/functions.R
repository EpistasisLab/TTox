# !/usr/bin/env Rscript
# created by Yun Hao @MooreLab 2019
# This script contains R functions required for other scripts in the repository.


## This function generates all possible combinations for a list of hyperparameters 
generate.all.possible.hyperparameter.combinations <- function(hp_list){
	## 0. Input arguments 
		# hp_list: list of hyperparameters   

	## 1. Generate all possible hyperparameter combinations
	# iterate by hyperparameters 
	hp_current <- hp_list[[1]];
	for (i in 2:length(hp_list)){
		# next hyperparameter
		hp_next <- hp_list[[i]];
		# combine two hyperparmeters  
		hp_current_vec <- rep(hp_current, each = length(hp_next));
		hp_next_vec <- rep(hp_next, times = length(hp_current)); 
		hp_current <- mapply(function(hcv, hnv) paste(hcv, hnv, sep = " "), hp_current_vec, hp_next_vec)
	}
	
	return(hp_current)
}


## This function generates executable shell scripts that will run input commands 
generate.parallel.bash.files <- function(all_commands, N_group, job_name, folder){
	## 0. Input arguments 
		# all_commands: a vector of commands that are to be run
		# N_group: number of groups that commands will be split into  
		# job_name: name of job 
		# folder: name of folder where shell scripts will be written 

	## 1. Assign the indices of commands to each group 
	# obtain the number of commands in each group
	N_group_member <- ceiling(length(all_commands) / N_group);
	# obtain upper bound of index for each group 
	upper_bound <- 1:N_group * N_group_member;
	ub_id <- min(which(upper_bound >= length(all_commands)));
	upper_bound <- upper_bound[1:ub_id];
	upper_bound[[ub_id]] <- length(all_commands);
	# obtain lower bound of index for each group 
	lower_bound <- 0:(ub_id-1) * N_group_member + 1;
	# assign commands to each group (lower bound - upper bound)
	command_list <- mapply(function(lb, ub) all_commands[lb:ub], lower_bound, upper_bound, SIMPLIFY = F);
	
	## 2. write commands into executable shell scripts
	# name executable shell scripts by "job_nameX.sh" (X is the index of script)
	setwd(folder);
	c_file_name <- sapply(1:ub_id, function(gn) paste(job_name, gn, ".sh", sep = ""));
	# iterate by script
	write_sub_files <- mapply(function(cl, cfn){
		# write commands into the script
		writeLines(cl, cfn);
		# make the script executable
		system(paste("chmod", "775", cfn, sep=" "));
		return(1);
	},command_list,c_file_name,SIMPLIFY=F);
	
	## 3. write an executable shell script that runs all the scripts above  
	final_command <- sapply(c_file_name,function(cfn) paste("./", folder, cfn," &", sep = ""));
	# name executable shell scripts by "job_name.sh" 
	final_file <- paste(job_name, ".sh", sep = "");
	writeLines(final_command, final_file);
	# make the script executable
	system(paste("chmod","775", final_file, sep = " "));

	return("DONE");
}


## This function groups a vector by categories of its elements.
group.vector.by.categories <- function(cate, vec){
	# 0. Input arguments 
		# cate: category of vectors  
		# vec: vector
	
	# 1. Sort vector by order of categories
	vec_od <- order(cate);
	cate <- cate[vec_od];
	vec <- vec[vec_od];

	# 2. Group elements of same category together
	# obtain unique categories
	cate_table <- table(cate);
	# obtain lower bound index/upper bound index of each unique category
	lower_ids <- cumsum(cate_table);
	upper_ids <- c(0, lower_ids[-length(lower_ids)]) + 1;
	# return list of vectors 
	vec_list <- mapply(function(li, ui) vec[li:ui], lower_ids, upper_ids, SIMPLIFY=F);
	names(vec_list) <- names(cate_table);

	return(vec_list);
}


## This function reads model performance metrics from files
read.performance.files <- function(perf_files, perf_hyper, hyper_names){
	## 0. Input arguments 
		# perf_files: model performance files derived from one data file 
		# perf_hyper: hyperparameter settings of model performance files 
		# hyper_names: all hyperparameter settings

	## 1. Read in all performance metric
	# iterate by performance file 
	all_perf_metric <- lapply(perf_files, function(pf){
		# read in all the lines from a performance file 
		metric_lines <- readLines(pf);
		# extract metric from each line 
		metrics <- sapply(metric_lines, function(ml){
			ml_s <-  strsplit(ml, ": ", fixed = T)[[1]];
			if(length(ml_s) < 2)	return('')
			else	return(ml_s[[2]])
		});
		return(metrics);
	});

	## 2. Obtain metrics of model using all features 
	all_feat_metric <- as.numeric(all_perf_metric[[1]][1:4]);
	names(all_feat_metric) <- c("N_train", "N_test", "N_all_features", "all_features_testing");
		 	
	## 3. Obtain number of features selected 
	N_select_feat <- sapply(all_perf_metric, function(apm) as.integer(apm[[6]]));	

	## 4. Obtain training performance of of feature selection 
	select_features_train <- sapply(all_perf_metric, function(apm){
		# obtain performance of multiple runs
		train_runs <- strsplit(apm[[7]], ",", fixed = T)[[1]];
		train_runs <- as.numeric(train_runs);
		# compute average of multiple runs 
		ave_train_runs <- mean(train_runs, na.rm = T);
		return(ave_train_runs);
	});		
	
	## 5. Obtain testing performance of feature selection
	select_features_test <- sapply(all_perf_metric, function(apm) as.numeric(apm[[8]])); 	

	## 6. Output vectors 
	N_select_feat_vec <- select_features_train_vec <- select_features_test_vec <- rep(NA, length(hyper_names));
	names(N_select_feat_vec) <- names(select_features_train_vec) <- names(select_features_test_vec) <- hyper_names;
	N_select_feat_vec[perf_hyper] <- N_select_feat;
	select_features_train_vec[perf_hyper] <- select_features_train;
	select_features_test_vec[perf_hyper] <- select_features_test;
	
	return(ls = list(all_features = all_feat_metric, select_features_N = N_select_feat_vec, select_features_train = select_features_train_vec, select_features_test = select_features_test_vec));
}
