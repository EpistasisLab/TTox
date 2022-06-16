# !/usr/bin/env python
## created by Yun Hao @MooreLab 2021
## This script shuffle outcomes labels of input feature-response data


## Module
import sys
sys.path.insert(0, 'src/')
import ttox_data


## Main function 
def main(argv):
	## 0. Input arguments: 
		# argv 1: input file that contains training feature-response data 
		# argv 2: name of label (response) column
		# argv 3: number of shuffled data files to be generated   
		# argv 4: prefix of output file name 

	## 1. Shuffle outcome labels of input feature-response data for specified times, write shuffled data to output folder 
	for i in range(0, int(argv[3])):
		shuffle_data = ttox_data.shuffle_data_outcome(argv[1], argv[2], i+1, argv[4])

	return 1


## Call main function
main(sys.argv)

