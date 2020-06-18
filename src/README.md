# This folder contains source code used by the repository.

## R/python scripts 

+ [`generate_learning_data.py`](generate_learning_data.py) generates feature-response datasets for learning task from group-sample relationships, then splits each dataset into train set and test set   

+ [`ttox_data.py`](ttox_data.py) contains functions for data pre-processing.  

## Executable shell scripts

+ [`run/generate_learning_data_target.sh`](run/generate_learning_data_target.sh) runs [`generate_learning_data.py`](generate_learning_data.py) on compound-target interactions data from BindingDB. [`run/generate_learning_data_toxicity.sh`](run/generate_learning_data_toxicity.sh) runs [`generate_learning_data.py`](generate_learning_data.py) on compound-toxicity data from OFFSIDES.

