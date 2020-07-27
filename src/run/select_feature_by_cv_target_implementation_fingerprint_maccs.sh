#!/bin/bash
#BSUB -q epistasis_long
#BSUB -J select_feature_by_cv_target_implementation_fingerprint_maccs
#BSUB -n 30
#BSUB -o src/run/select_feature_by_cv_target_implementation_fingerprint_maccs.%J.out
#BSUB -e src/run/select_feature_by_cv_target_implementation_fingerprint_maccs.%J.error
#BSUB -N

module unload python
module load python/3.8

./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs1.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs2.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs3.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs4.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs5.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs6.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs7.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs8.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs9.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs10.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs11.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs12.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs13.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs14.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs15.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs16.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs17.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs18.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs19.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs20.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs21.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs22.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs23.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs24.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs25.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs26.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs27.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs28.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs29.sh &
./src/run/select_feature_by_cv_target_implementation_fingerprint_maccs30.sh &
