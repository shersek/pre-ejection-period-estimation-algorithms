#!/bin/bash

python3 hyper_parameter_sweep.py \
--csv_file '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Data Analysis for Gyroscope Accelerometer Data/SCG Gyro Project Rest Rec Features 16 Subjects 20170729.csv' \
--N_repetitions 50 \
--axis_combo 'gX,gY,aX,aZ' \
--hyper_parameter_to_sweep 'lasso_alpha' \
--number_of_points 200


python3 hyper_parameter_sweep.py \
--csv_file '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Data Analysis for Gyroscope Accelerometer Data/SCG Gyro Project Rest Rec Features 16 Subjects 20170729.csv' \
--N_repetitions 50 \
--axis_combo 'gX,gY,aX,aZ' \
--hyper_parameter_to_sweep 'ridge_alpha' \
--number_of_points 200


python3 hyper_parameter_sweep.py \
--csv_file '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Data Analysis for Gyroscope Accelerometer Data/SCG Gyro Project Rest Rec Features 16 Subjects 20170729.csv' \
--xgb_max_depth 5 \
--xgb_subsample 0.5 \
--xgb_colsample_bytree 0.5 \
--xgb_num_boost_round 200 \
--N_repetitions 50 \
--ridge_alpha 1 \
--lasso_alpha 1 \
--axis_combo 'gX,gY,aX,aZ' \
--hyper_parameter_to_sweep 'xgb_eta' \
--number_of_points 200

python3 hyper_parameter_sweep.py \
--csv_file '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Data Analysis for Gyroscope Accelerometer Data/SCG Gyro Project Rest Rec Features 16 Subjects 20170729.csv' \
--xgb_eta 0.1 \
--xgb_max_depth 5 \
--xgb_subsample 0.5 \
--xgb_num_boost_round 200 \
--N_repetitions 50 \
--ridge_alpha 1 \
--lasso_alpha 1 \
--axis_combo 'gX,gY,aX,aZ' \
--hyper_parameter_to_sweep 'xgb_colsample_bytree' \
--number_of_points 200


python3 hyper_parameter_sweep.py \
--csv_file '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Data Analysis for Gyroscope Accelerometer Data/SCG Gyro Project Rest Rec Features 16 Subjects 20170729.csv' \
--xgb_eta 0.1 \
--xgb_max_depth 5 \
--xgb_colsample_bytree 0.5 \
--xgb_num_boost_round 200 \
--N_repetitions 50 \
--ridge_alpha 1 \
--lasso_alpha 1 \
--axis_combo 'gX,gY,aX,aZ' \
--hyper_parameter_to_sweep 'xgb_subsample' \
--number_of_points 200


