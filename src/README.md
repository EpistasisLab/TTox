# This folder contains source code used by the repository.

## R/python scripts 

+ [`generate_learning_data.py`](generate_learning_data.py) generates feature-response datasets for learning task from group-sample relationships, then splits each dataset into train set and test set   

+ [`ttox_data.py`](ttox_data.py) contains functions for data pre-processing.  

+ [`compare_train_test_similarity.py`](compare_train_test_similarity.py) computes and visualizes structure similarity between compounds.

+ [`ttox_sim.py`](ttox_sim.py) contains functions for computing structure similarity between compounds, and feature similarity between models.

+ [`select_feature_by_cv.py`](select_feature_by_cv.py) combines ReBATE methods and cross-validation to select relevant features, then evaluates model performance on hold-out testing set, eventually implements the model to predict responses of new instances. [`run/run_select_feature_by_cv_target_tuning.R`](run/run_select_feature_by_cv_target_tuning.R) generates shell scripts that run feature selection hyperparamter tuning on sampled subset of compound-target interaction datasets. [`run/run_select_feature_by_cv_target_implementation.R`](run/run_select_feature_by_cv_target_implementation.R) generates shell scripts that implement feature selection on all compound-target interaction datasets using the optimal hyperparameter settings. 

+ [`ttox_learning.py`](ttox_learning.py) contains functions for building, evaluating, and implementing machine learning models. 

+ [`ttox_selection.py`](ttox_selection.py) contains functions for selecting relevant features by cross-validation. 

+ [`collect_selection_results.R`](collect_selection_results.R) collects feature selection results from multiple datasets.

+ [`functions.R`](functions.R) R functions required for other R scripts in the repository.

+ [`analyze_selection_results.py`](analyze_selection_results.py) analyzes and visualizes the results of feature selection pipeline.  

+ [`ttox_plot.py`](ttox_plot.py) contains functions for visualizing feature selection results. 

+ [`compare_feature_similarity.py`](compare_feature_similarity.py) analyzes similarity of selected relevant features among different models. 

## Executable shell scripts

+ [`run/generate_learning_data_target.sh`](run/generate_learning_data_target.sh) runs [`generate_learning_data.py`](generate_learning_data.py) on compound-target interactions data from BindingDB. [`run/generate_learning_data_toxicity.sh`](run/generate_learning_data_toxicity.sh) runs [`generate_learning_data.py`](generate_learning_data.py) on compound-toxicity data from OFFSIDES.

+ [`run/compare_train_test_similarity_offsides.sh`](run/compare_train_test_similarity_offsides.sh) runs [`compare_train_test_similarity.py`](compare_train_test_similarity.py) on generated structure-toxicity data of OFFSIDES compounds.

+ [`run/run_select_feature_by_cv_target_tuning.sh`](run/run_select_feature_by_cv_target_tuning.sh) runs [`run/run_select_feature_by_cv_target_tuning.R`](run/run_select_feature_by_cv_target_tuning.R) on two types of compound-target interaction datasets: chemical fingerprints and molecular descriptors. [`run/run_select_feature_by_cv_target_implementation.sh`](run/run_select_feature_by_cv_target_implementation.sh) runs [`run/run_select_feature_by_cv_target_implementation.R`](run/run_select_feature_by_cv_target_implementation.R) on two types of compound-target interaction datasets: chemical fingerprints and molecular descriptors.

+ [`run/select_feature_by_cv_target_tuning_fingerprint_maccs.sh`](run/select_feature_by_cv_target_tuning_fingerprint_maccs.sh) and [`run/select_feature_by_cv_target_tuning_descriptor_all.sh`](run/select_feature_by_cv_target_tuning_descriptor_all.sh) runs [`select_feature_by_cv.py`](select_feature_by_cv.py) on two types of compound-target interaction datasets: chemical fingerprints and molecular descriptors, respectively, for the purpose of feature selection hyperparamter tuning. [`run/select_feature_by_cv_target_implementation_fingerprint_maccs.sh`](run/select_feature_by_cv_target_implementation_fingerprint_maccs.sh) and [`run/select_feature_by_cv_target_implementation_descriptor_all.sh`](run/select_feature_by_cv_target_implementation_descriptor_all.sh) on two types of compound-target interaction datasets: chemical fingerprints and molecular descriptors, respectively, for the purpose of feature selection implementation.

+ [`run/collect_selection_results_target_tuning.sh`](run/collect_selection_results_target_tuning.sh) runs [`collect_selection_results.R`](collect_selection_results.R) on compound-target interaction datasets selected for hyperparamter tuning. [`run/collect_selection_results_target_implementation.sh`](run/collect_selection_results_target_implementation.sh) runs [`collect_selection_results.R`](collect_selection_results.R) on all compound-target interaction datasets for feature selection pipeline implementation. 

+ [`run/analyze_selection_results_target_tuning.sh`](run/analyze_selection_results_target_tuning.sh) runs [`analyze_selection_results.py`](analyze_selection_results.py) on feature selection hyperparamter tuning results of compound-target interaction datasets. [`run/analyze_selection_results_target_implementation.sh`](run/analyze_selection_results_target_implementation.sh) runs [`analyze_selection_results.py`](analyze_selection_results.py) on feature selection pipeline implementation results of all compound-target interaction datasets.

+ [`run/compare_feature_similarity_structure.sh`](run/compare_feature_similarity_structure.sh) runs [`compare_feature_similarity.py`](compare_feature_similarity.py) on feature selection pipeline implementation results of all compound-target interaction datasets. 
