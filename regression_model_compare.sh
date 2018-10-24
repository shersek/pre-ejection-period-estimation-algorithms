#!/bin/bash

for i in 'gX' 'gY' 'gZ' 'aX' 'aY' 'aZ' 'gX,gY' 'gX,gZ' 'gY,gZ' 'gX,gY,gZ' 'aX,aY' 'aX,aZ' 'aY,aZ' 'aX,aY,aZ' 'gX,gY,aX,aZ' 'gX,gY,gZ,aX,aY,aZ'
	do  

	python3 regression_model_compare.py \
	--csv_file '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Data Analysis for Gyroscope Accelerometer Data/SCG Gyro Project Rest Rec Features 16 Subjects 20170729.csv' \
	--xgb_eta 0.1 \
	--xgb_max_depth 5 \
	--xgb_subsample 0.5 \
	--xgb_colsample_bytree 0.5 \
	--xgb_num_boost_round 200 \
	--rfr_n_estimators 200 \
	--rfr_max_depth 10 \
	--et_max_depth 10 \
	--et_n_estimators 200 \
	--ridge_alpha 1 \
	--lasso_alpha 1 \
	--axis_combo $i \
	--N_repetitions 5

	done

