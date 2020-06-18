# This folder contains source code used by the repository.

## R/python scripts 

+ [`generate_learning_data.py`](generate_learning_data.py) generates feature-response datasets for learning task from group-sample relationships, then splits each dataset into train set and test set   

+ [`ttox_data.py`](ttox_data.py) contains functions for data pre-processing.  

+ [`compare_train_test_similarity.py`](compare_train_test_similarity.py) computes and visualizes structure similarity between compounds.

+ [`ttox_sim.py`](ttox_sim.py) contains functions for computing structure similarity between compounds.  

## Executable shell scripts

+ [`run/generate_learning_data_target.sh`](run/generate_learning_data_target.sh) runs [`generate_learning_data.py`](generate_learning_data.py) on compound-target interactions data from BindingDB. [`run/generate_learning_data_toxicity.sh`](run/generate_learning_data_toxicity.sh) runs [`generate_learning_data.py`](generate_learning_data.py) on compound-toxicity data from OFFSIDES.

+ [`run/compare_train_test_similarity_offsides.sh`](run/compare_train_test_similarity_offsides.sh) runs [`compare_train_test_similarity.py`](compare_train_test_similarity.py) on generated structure-toxicity data of OFFSIDES compounds.
