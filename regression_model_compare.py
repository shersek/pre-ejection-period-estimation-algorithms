import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean
import utility
import argparse

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--xgb_eta', default=0.1, type=float, help='Eta for the xgboost regressor')
    arg('--xgb_max_depth', default=5, type=int , help='Max depth for the xgboost regressor')
    arg('--xgb_subsample', default=0.5, type=float , help='Subsample rate for the xgboost regressor' )
    arg('--xgb_colsample_bytree', default=0.5, type=float , help='Feature subsampling rate for the xgboost regressor')
    arg('--xgb_num_boost_round', default=10, type=int , help='Number of boosting rounds for the xgboost regressor')

    arg('--rfr_max_depth', default=5, type=int , help='Max depth for the random forest regressor')
    arg('--rfr_n_estimators', default=10, type=int , help='Number of trees for the random forest regressor')

    arg('--et_max_depth', default=5, type=int , help='Max depth for the extra trees regressor')
    arg('--et_n_estimators', default=10, type=int , help='Number of trees for the extra trees regressor')

    arg('--ridge_alpha', default=1, type=float , help='Alpha for ridge regression')

    arg('--lasso_alpha', default=1, type=float , help='Alpha for lasso regression')

    arg('--N_repetitions', type=int , help='Number of cross-validation repetitions')

    arg('--csv_file', type=str , help='File containing all features extracted from the signals')

    arg('--axis_combo', type=str , help='Sensors and axis to use, e.g. use only accelerometer z axis: az, use a combination of accelerometer Z and gyroscope X axes: aZ,gX ')

    args = vars(parser.parse_args())

    #read dataset
    data_frame = pd.read_csv(args['csv_file'])

    #get rid of NaN's due to feature extraction
    data_frame=data_frame.dropna(axis= 'index' , how='any')

    #get rid of very high PEP values (outliers)
    data_frame=data_frame[data_frame['PEP']<200]

    #set up experiment: number of rounds CV is repeated and the feature set
    N_repetitions= args['N_repetitions']

    #features to use
    axis_combination = args['axis_combo'].split(',')

    feature_set = []
    if 'gX' in axis_combination:
        feature_set += utility.get_gyro_x_features()
    if 'gY' in axis_combination:
        feature_set += utility.get_gyro_y_features()
    if 'gZ' in axis_combination:
        feature_set += utility.get_gyro_z_features()
    if 'aX' in axis_combination:
        feature_set += utility.get_acc_x_features()
    if 'aY' in axis_combination:
        feature_set += utility.get_acc_y_features()
    if 'aZ' in axis_combination:
        feature_set += utility.get_acc_z_features()

    #models to test
    list_of_models = [

    utility.RegressionModel(feature_set=feature_set, name='xgb model' , type='xgb' ,
    hyper_parameters={
        'eta': args['xgb_eta'],
        'max_depth': args['xgb_max_depth'],
        'subsample': args['xgb_subsample'],
        'objective': 'reg:linear',
        'colsample_bytree': args['xgb_colsample_bytree'],
        'num_boost_round':args['xgb_num_boost_round']

    })    ,

    utility.RegressionModel(feature_set=feature_set, name='rfr model', type='rfr',
                            hyper_parameters={
                                'max_depth': args['rfr_max_depth'],
                                'max_features':'sqrt',
                                'n_estimators':args['rfr_n_estimators']
                            }),

    utility.RegressionModel(feature_set=feature_set, name='et model', type='et',
                            hyper_parameters={
                                'max_depth': args['et_max_depth'],
                                'max_features':'sqrt',
                                'n_estimators':args['et_n_estimators']
                            }),


    utility.RegressionModel(feature_set=feature_set, name='linreg model', type='linr'),

    utility.RegressionModel(feature_set=feature_set, name='ridge', type='ridr',
                            hyper_parameters={
                                'alpha': args['ridge_alpha'],
                            }),

    utility.RegressionModel(feature_set=feature_set, name='lasso', type='lasr',
                            hyper_parameters={
                                'alpha': args['lasso_alpha'],
                            })
    ]

    #run experiments
    scores = []
    list_regr_name = []
    list_model_type = []
    for regr_model in list_of_models:
        score_vector_rmse = utility.cross_validate_rmse(data_frame, N_repetitions , regr_model)
        for score in score_vector_rmse:
            scores.append(score)
            list_regr_name.append(regr_model.name)
            list_model_type.append('Non-linear' if regr_model.type not in ['linr' , 'ridr', 'lasr'] else 'Linear')

    #add results to a data frame
    dfr_rmse_results = pd.DataFrame(data={'rmse': scores , 'Regressor':list_regr_name , 'Regression Type':list_model_type})

    #construct barplot
    fig = plt.figure(figsize=(12,8));
    sns.set_style('whitegrid', {'grid.linestyle':'--'})
    ax=sns.barplot(x='Regressor', y='rmse', hue = 'Regression Type' , data=dfr_rmse_results , linewidth=1.5 ,
                   estimator=mean , ci='sd' , dodge = False ,
                   palette=sns.xkcd_palette(['cherry red' , 'cerulean blue']) , errcolor ='.2', edgecolor='.2'
                   , alpha=0.8, capsize=0.2)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    plt.title(args['axis_combo'])
    plt.ylim(0,22)
    fig.savefig(args['axis_combo'], dpi=fig.dpi)


if __name__ == '__main__':
    main()