#!/bin/bash
#BSUB -q epistasis_long
#BSUB -J select_feature_by_cv_target_tuning_descriptor_all
#BSUB -n 30
#BSUB -o src/run/select_feature_by_cv_target_tuning_descriptor_all.%J.out
#BSUB -e src/run/select_feature_by_cv_target_tuning_descriptor_all.%J.error
#BSUB -N

module unload python
module load python/3.8

./src/run/select_feature_by_cv_target_tuning_descriptor_all1.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all2.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all3.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all4.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all5.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all6.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all7.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all8.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all9.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all10.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all11.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all12.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all13.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all14.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all15.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all16.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all17.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all18.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all19.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all20.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all21.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all22.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all23.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all24.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all25.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all26.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all27.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all28.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all29.sh &
./src/run/select_feature_by_cv_target_tuning_descriptor_all30.sh &
