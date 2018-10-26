import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import utility
import argparse


def main():

    def plot_hyper_parameter_sweep(list_of_models, sweep):
        mean_scores = []
        std_scores = []
        for regr_model in list_of_models:
            score_vector_rmse = utility.cross_validate_rmse(data_frame, N_repetitions, regr_model)
            mean_scores.append(np.mean(score_vector_rmse))
            std_scores.append(np.std(score_vector_rmse))

        fig = plt.figure()
        plt.fill_between(sweep, np.array(mean_scores) - np.array(std_scores),
                         np.array(mean_scores) + np.array(std_scores), facecolor='xkcd:light pink', alpha=0.7)
        plt.semilogx(sweep, mean_scores, color='xkcd:red', linewidth=4)
        plt.xlabel('Hyper-parameter')
        plt.ylabel('RMSE')
        plt.grid(True, which='both')
        plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
        plt.tight_layout()
        fig.savefig(args['hyper_parameter_to_sweep'], dpi=200)


    #config
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--xgb_eta', default=0.1, type=float, help='Eta for the xgboost regressor')
    arg('--xgb_max_depth', default=5, type=int , help='Max depth for the xgboost regressor')
    arg('--xgb_subsample', default=0.5, type=float , help='Subsample rate for the xgboost regressor' )
    arg('--xgb_colsample_bytree', default=0.5, type=float , help='Feature subsampling rate for the xgboost regressor')
    arg('--xgb_num_boost_round', default=10, type=int , help='Number of boosting rounds for the xgboost regressor')
    arg('--ridge_alpha', default=1, type=float , help='Alpha for ridge regression')
    arg('--lasso_alpha', default=1, type=float , help='Alpha for lasso regression')
    arg('--N_repetitions', type=int , help='Number of cross-validation repetitions')
    arg('--csv_file', type=str , help='File containing all features extracted from the signals')
    arg('--axis_combo', type=str , help='Sensors and axis to use, e.g. use only accelerometer z axis: az, use a combination of accelerometer Z and gyroscope X axes: aZ,gX ')
    arg('--hyper_parameter_to_sweep', type=str , help='Options are: xgb_eta, xgb_max_depth, xgb_subsample, xgb_colsample_bytree, ridge_alpha, lasso_alpha')
    arg('--number_of_points', type=str, help='Number of points on the sweep grid')
    args = vars(parser.parse_args())


    #close all pre-existing plots and modify plot settings
    plt.close('all')
    plt.interactive(False)

    #read dataset
    data_frame =     data_frame = pd.read_csv(args['csv_file'])

    #there are many Nan's in the data, get rid of them
    data_frame=data_frame.dropna(axis= 'index' , how='any')

    #get rid of very high PEP values
    data_frame=data_frame[data_frame['PEP']<200]

    #set up experiment: number of rounds CV is repeated and the feature set
    N_repetitions= args['N_repetitions']

    #form feature set
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


    if args['hyper_parameter_to_sweep'] == 'lasso_alpha':

        list_of_models = []
        sweep =   np.power(10, np.linspace(-3,2,args['number_of_points']))
        for alpha in sweep:
            list_of_models.append( utility.RegressionModel( feature_set=feature_set, name = str(alpha) , type='lasr' , hyper_parameters= {'alpha':alpha} ) )

        plot_hyper_parameter_sweep(list_of_models, sweep)
    
    elif args['hyper_parameter_to_sweep'] == 'ridge_alpha':

        list_of_models = []
        sweep =   np.power(10, np.linspace(-3,2,args['number_of_points']))
        for alpha in sweep:
            list_of_models.append( utility.RegressionModel( feature_set=feature_set, name = str(alpha) , type='ridr' , hyper_parameters= {'alpha':alpha} ) )

        plot_hyper_parameter_sweep(list_of_models, sweep)


    elif args['hyper_parameter_to_sweep'] == 'xgb_eta':

        list_of_models = []
        sweep = np.power(10, np.linspace(-3, 2, args['number_of_points']))
        for alpha in sweep:
            list_of_models.append(     utility.RegressionModel(feature_set=feature_set, name='xgb model' , type='xgb' ,
                                        hyper_parameters={
                                            'eta': alpha ,
                                            'max_depth': args['xgb_max_depth'],
                                            'subsample': args['xgb_subsample'],
                                            'objective': 'reg:linear',
                                            'colsample_bytree': args['xgb_colsample_bytree'],
                                            'num_boost_round':args['xgb_num_boost_round']

                                        })  )

        plot_hyper_parameter_sweep(list_of_models, sweep)

    elif args['hyper_parameter_to_sweep'] == 'xgb_subsample':

        list_of_models = []
        sweep = np.linspace(0.1, 0.9, args['number_of_points'])
        for alpha in sweep:
            list_of_models.append(     utility.RegressionModel(feature_set=feature_set, name='xgb model' , type='xgb' ,
                                        hyper_parameters={
                                            'eta': args['xgb_eta'] ,
                                            'max_depth': args['xgb_max_depth'],
                                            'subsample': alpha,
                                            'objective': 'reg:linear',
                                            'colsample_bytree': args['xgb_colsample_bytree'],
                                            'num_boost_round':args['xgb_num_boost_round']

                                        })  )

        plot_hyper_parameter_sweep(list_of_models, sweep)


    elif args['hyper_parameter_to_sweep'] == 'xgb_colsample_bytree':

        list_of_models = []
        sweep = np.linspace(0.1, 0.9, args['number_of_points'])
        for alpha in sweep:
            list_of_models.append(     utility.RegressionModel(feature_set=feature_set, name='xgb model' , type='xgb' ,
                                        hyper_parameters={
                                            'eta': args['xgb_eta'] ,
                                            'max_depth': args['xgb_max_depth'],
                                            'subsample': args['xgb_subsample'],
                                            'objective': 'reg:linear',
                                            'colsample_bytree': alpha ,
                                            'num_boost_round':args['xgb_num_boost_round']

                                        })  )

        plot_hyper_parameter_sweep(list_of_models, sweep)





if __name__ == '__main__':
    main()
