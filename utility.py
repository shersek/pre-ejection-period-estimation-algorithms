from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model
import numpy as np
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
import xgboost as xgb

class RegressionModel(object):
    """
        Instance class represents set of raw data collected per subject
    """

    def __init__(self, feature_set, name , type , hyper_parameters=None):
        self.feature_set = feature_set
        self.name =  name
        self.type = type
        self.hyper_parameters = hyper_parameters
        self.model = self._get_model()

    def _get_model(self ):


        if self.type == 'rfr':
            model = RandomForestRegressor(n_estimators=self.hyper_parameters['n_estimators']
                                          , max_depth=self.hyper_parameters['max_depth'],
                                          max_features=self.hyper_parameters['max_features'],
                                          random_state=2 )
        elif self.type == 'et':
            model = ExtraTreesRegressor(n_estimators=self.hyper_parameters['n_estimators']
                                          , max_depth=self.hyper_parameters['max_depth'],
                                          max_features=self.hyper_parameters['max_features'],
                                          random_state=2 )
        elif self.type == 'linr':
            model = linear_model.LinearRegression()
        elif self.type == 'ridr':
            model = linear_model.Ridge(alpha=self.hyper_parameters['alpha'], random_state=2)
        elif self.type == 'lasr':
            model = linear_model.Lasso(alpha=self.hyper_parameters['alpha'], random_state=2)
        else:
            model = None

        return model




def get_gyro_x_features():
    return ['Gyro X Location First Maxima 0-250 ms',
                    'Gyro X Width First Maxima 0-250 ms',
                    'Gyro X Location Second Maxima 0-250 ms',
                    'Gyro X Width Second Maxima 0-250 ms',
                    'Gyro X Location First Maxima 250-500 ms',
                    'Gyro X Width First Maxima 250-500 ms',
                    'Gyro X Location First Minima 0-250 ms',
                    'Gyro X Width First Minima 0-250 ms',
                    'Gyro X Location Second Minima 0-250 ms',
                    'Gyro X Width Second Minima 0-250 ms',
                    'Gyro X Location First Minima 250-500 ms',
                    'Gyro X Width First Minima 250-500 ms']

def get_gyro_y_features():

    return ['Gyro Y Location First Maxima 0-250 ms',
                    'Gyro Y Width First Maxima 0-250 ms',
                    'Gyro Y Location Second Maxima 0-250 ms',
                    'Gyro Y Width Second Maxima 0-250 ms',
                    'Gyro Y Location First Maxima 250-500 ms',
                    'Gyro Y Width First Maxima 250-500 ms',
                    'Gyro Y Location First Minima 0-250 ms',
                    'Gyro Y Width First Minima 0-250 ms',
                    'Gyro Y Location Second Minima 0-250 ms',
                    'Gyro Y Width Second Minima 0-250 ms',
                    'Gyro Y Location First Minima 250-500 ms',
                    'Gyro Y Width First Minima 250-500 ms']

def get_gyro_z_features():

    return['Gyro Z Location First Maxima 0-250 ms',
                    'Gyro Z Width First Maxima 0-250 ms',
                    'Gyro Z Location Second Maxima 0-250 ms',
                    'Gyro Z Width Second Maxima 0-250 ms',
                    'Gyro Z Location First Maxima 250-500 ms',
                    'Gyro Z Width First Maxima 250-500 ms',
                    'Gyro Z Location First Minima 0-250 ms',
                    'Gyro Z Width First Minima 0-250 ms',
                    'Gyro Z Location Second Minima 0-250 ms',
                    'Gyro Z Width Second Minima 0-250 ms',
                    'Gyro Z Location First Minima 250-500 ms',
                    'Gyro Z Width First Minima 250-500 ms']

def get_acc_x_features():
    return ['ACC X Location First Maxima 0-250 ms',
                    'ACC X Width First Maxima 0-250 ms',
                    'ACC X Location Second Maxima 0-250 ms',
                    'ACC X Width Second Maxima 0-250 ms',
                    'ACC X Location First Maxima 250-500 ms',
                    'ACC X Width First Maxima 250-500 ms',
                    'ACC X Location First Minima 0-250 ms',
                    'ACC X Width First Minima 0-250 ms',
                    'ACC X Location Second Minima 0-250 ms',
                    'ACC X Width Second Minima 0-250 ms',
                    'ACC X Location First Minima 250-500 ms',
                    'ACC X Width First Minima 250-500 ms']

def get_acc_y_features():
    return ['ACC Y Location First Maxima 0-250 ms',
                    'ACC Y Width First Maxima 0-250 ms',
                    'ACC Y Location Second Maxima 0-250 ms',
                    'ACC Y Width Second Maxima 0-250 ms',
                    'ACC Y Location First Maxima 250-500 ms',
                    'ACC Y Width First Maxima 250-500 ms',
                    'ACC Y Location First Minima 0-250 ms',
                    'ACC Y Width First Minima 0-250 ms',
                    'ACC Y Location Second Minima 0-250 ms',
                    'ACC Y Width Second Minima 0-250 ms',
                    'ACC Y Location First Minima 250-500 ms',
                    'ACC Y Width First Minima 250-500 ms']

def get_acc_z_features():
    return ['ACC Z Location First Maxima 0-250 ms',
                    'ACC Z Width First Maxima 0-250 ms',
                    'ACC Z Location Second Maxima 0-250 ms',
                    'ACC Z Width Second Maxima 0-250 ms',
                    'ACC Z Location First Maxima 250-500 ms',
                    'ACC Z Width First Maxima 250-500 ms',
                    'ACC Z Location First Minima 0-250 ms',
                    'ACC Z Width First Minima 0-250 ms',
                    'ACC Z Location Second Minima 0-250 ms',
                    'ACC Z Width Second Minima 0-250 ms',
                    'ACC Z Location First Minima 250-500 ms',
                    'ACC Z Width First Minima 250-500 ms']

def get_bcg_features():
    return [
               'R-I interval',
               'R-J Interval',
               'R-K Interval'
               ];



#seperate the subjects into random groups of two
def seperate_subjects_into_cv_groups(subjectIDs , rnd):
    #shuffle subjects
    uniqueSubjectIDs = np.unique(subjectIDs)
    rnd.shuffle(uniqueSubjectIDs)
    print(uniqueSubjectIDs)
    uniqueGroups = np.repeat(np.arange(uniqueSubjectIDs.shape[0]/2), 2).astype(int)

    #assign subjects to groups
    groups = np.zeros(subjectIDs.shape)
    indexGroup = 0
    for u in uniqueSubjectIDs:
        groups[subjectIDs==u] = uniqueGroups[indexGroup]
        indexGroup = indexGroup +1

    return groups

#function to evaluate a feature set using xgboost and repeated CV
def cross_validate_rmse(data_frame, N_repetitions , regr_model_instance):

    #get the data set
    X = data_frame[regr_model_instance.feature_set].values
    y = data_frame['PEP'].values
    subjectIDs = (data_frame['Subject ID'].values).astype(int)

    #perform smoothing to the target variable: PEP for each subject
    for subjectNo in np.unique(subjectIDs):

        #get PEP for the subject
        pep = y[subjectIDs==subjectNo]
        tt = np.arange(0,pep.shape[0])

        #smooth PEP
        pep_smoothed=signal.medfilt(pep, 9)
        y[subjectIDs==subjectNo] = pep_smoothed

    #initialize a vector to collect all "out of fold" predictions
    y_all_predictions = np.zeros(y.shape)

    #set up an array to keep all cv errors from the repetitions
    score_vector_rmse = []

    #fix random seed to get consistent shuffles in each experiment
    rnd = np.random.RandomState(42) #fix random seed !

    for reps in np.arange(N_repetitions):

        #groups subjects into pairs randomly for CV grouping and set up CV
        print(reps)
        groups_cv = seperate_subjects_into_cv_groups(subjectIDs , rnd)
        logo = LeaveOneGroupOut()

        #cross validation, leaving two subjects out
        for train, test in logo.split(X, y, groups=groups_cv):

            #perform train test split for the current fold
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            #apply explonential transformation to PEP
            y_train_transformed = np.exp(y_train/100)


            if regr_model_instance.type =='xgb':
                #define regression model
                xgb_params={
                    'eta': regr_model_instance.hyper_parameters['eta'],
                    'max_depth': regr_model_instance.hyper_parameters['max_depth'],
                    'subsample': regr_model_instance.hyper_parameters['subsample'],
                    'objective': 'reg:linear',
                     'colsample_bytree': regr_model_instance.hyper_parameters['colsample_bytree'],
                    'eval_metric': 'rmse',
                    'base_score': np.exp(120/100), # base prediction = mean(target)
                    'silent': 1
                }

                #reformat training set, note that I estimate the exp transformed PEP
                dtrain = xgb.DMatrix(X_train, y_train_transformed)

                #train model
                xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=regr_model_instance.hyper_parameters['num_boost_round'])

                #predict testing samples using trained model
                dtest = xgb.DMatrix(X_test)
                y_predicted_transformed = xgb_model.predict(dtest)

            else:
                # train model
                model_trained = regr_model_instance.model.fit(X_train, y_train_transformed)

                # predict testing samples using trained model
                y_predicted_transformed = model_trained.predict(X_test)

            #transform back
            y_predicted = 100 * np.log(y_predicted_transformed)  # inverse transform

            #smooth the prediction output
            y_predicted=signal.medfilt(y_predicted, 9)

            #add predictions to the out of fold predictions vector
            y_all_predictions[test] = y_predicted

        #calculate rmse for this repetition
        rmse = np.sqrt(mean_squared_error(y, y_all_predictions))

        #add the rmse from this repetition to a vector
        score_vector_rmse.append(rmse)

    #print results
    print(regr_model_instance.name)
    print('RMSE= ' + str(np.mean(np.asarray(score_vector_rmse))) + ' +/- ' + str(np.std(np.asarray(score_vector_rmse))))

    return score_vector_rmse



