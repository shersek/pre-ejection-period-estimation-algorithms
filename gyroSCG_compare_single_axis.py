import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import seaborn as sns
from numpy import mean
from sklearn import linear_model
import utility

#close all pre-existing plots and modify plot settings
plt.close('all')
plt.interactive(False)

#read dataset
data_frame = pd.read_csv('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Data Analysis for Gyroscope Accelerometer Data/SCG Gyro Project Rest Rec Features 16 Subjects 20170729.csv')

#there are many Nan's in the data, get rid of them
data_frame=data_frame.dropna(axis= 'index' , how='any')

#get rid of very high PEP values
data_frame=data_frame[data_frame['PEP']<200]

#set up experiment: number of rounds CV is repeated and the feature set
N_repetitions= 3
feature_set = []
feature_set+=utility.get_gyro_x_features()
feature_set+=utility.get_gyro_y_features()
feature_set+=utility.get_acc_x_features()
feature_set+=utility.get_acc_z_features()

list_of_models = [

utility.RegressionModel(feature_set=feature_set, name='xgb model' , type='xgb' ,
hyper_parameters={
    'eta': 0.1,
    'max_depth': 5,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'colsample_bytree': 0.5,
    'num_boost_round':200

})    ,

utility.RegressionModel(feature_set=feature_set, name='rfr model', type='rfr',
                        hyper_parameters={
                            'max_depth': 5,
                            'max_features':'sqrt',
                            'n_estimators':200
                        }),

utility.RegressionModel(feature_set=feature_set, name='et model', type='et',
                        hyper_parameters={
                            'max_depth': 5,
                            'max_features':'sqrt',
                            'n_estimators':200
                        }),


utility.RegressionModel(feature_set=feature_set, name='linreg model', type='linr'),

utility.RegressionModel(feature_set=feature_set, name='ridge', type='ridr',
                        hyper_parameters={
                            'alpha': 1,
                        }),

utility.RegressionModel(feature_set=feature_set, name='lasso', type='lasr',
                        hyper_parameters={
                            'alpha': 1,
                        })
]

scores = []
list_regr_name = []
list_model_type = []
for regr_model in list_of_models:
    score_vector_rmse = utility.cross_validate_rmse(data_frame, N_repetitions , regr_model)
    for score in score_vector_rmse:
        scores.append(score)
        list_regr_name.append(regr_model.name)
        list_model_type.append('Non-linear' if regr_model.type not in ['linr' , 'ridr', 'lasr'] else 'Linear')

dfr_rmse_results = pd.DataFrame(data={'rmse': scores , 'Regressor':list_regr_name , 'Regression Type':list_model_type})


#construct barplot
fig = plt.figure(figsize=(4.5,2.9));
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
plt.ylim(0,18)
# import matplotlib.ticker as ticker
# ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
plt.show()

