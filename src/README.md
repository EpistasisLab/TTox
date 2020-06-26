# This folder contains source code used by the repository.

## R/python scripts 

+ [`generate_learning_data.py`](generate_learning_data.py) generates feature-response datasets for learning task from group-sample relationships, then splits each dataset into train set and test set   

+ [`ttox_data.py`](ttox_data.py) contains functions for data pre-processing.  

+ [`compare_train_test_similarity.py`](compare_train_test_similarity.py) computes and visualizes structure similarity between compounds.

+ [`ttox_sim.py`](ttox_sim.py) contains functions for computing structure similarity between compounds.  

+ [`select_feature_by_cv.py`](select_feature_by_cv.py) combines ReBATE methods and cross-validation to select relevant features, then evaluates model performance on hold-out testing set, eventually implements the model to predict responses of new instances. [`run/run_select_feature_by_cv_target.R`](run/run_select_feature_by_cv_target.R) generates shell scripts that run the pipeline on compound-target interaction datasets.

+ [`ttox_learning.py`](ttox_learning.py) contains functions for building, evaluating, and implementing machine learning models. 

+ [`ttox_selection.py`](ttox_selection.py) contains functions for selecting relevant features by cross-validation. 

+ [`functions.R`](functions.R) R functions required for other R scripts in the repository.

## Executable shell scripts

+ [`run/generate_learning_data_target.sh`](run/generate_learning_data_target.sh) runs [`generate_learning_data.py`](generate_learning_data.py) on compound-target interactions data from BindingDB. [`run/generate_learning_data_toxicity.sh`](run/generate_learning_data_toxicity.sh) runs [`generate_learning_data.py`](generate_learning_data.py) on compound-toxicity data from OFFSIDES.

+ [`run/compare_train_test_similarity_offsides.sh`](run/compare_train_test_similarity_offsides.sh) runs [`compare_train_test_similarity.py`](compare_train_test_similarity.py) on generated structure-toxicity data of OFFSIDES compounds.

+ [`run/select_feature_by_cv_target.sh`](run/select_feature_by_cv_target.sh) runs [`select_feature_by_cv.py`](select_feature_by_cv.py) on compound-target interaction datasets. 
