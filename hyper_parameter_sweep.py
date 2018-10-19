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

list_of_models = []
alpha_sweep =   np.power(10, np.linspace(-3,2,1000))
for alpha in alpha_sweep:
    list_of_models.append( utility.RegressionModel( feature_set=feature_set, name = str(alpha) , type='lasr' , hyper_parameters= {'alpha':alpha} ) )

mean_scores = []
std_scores =[]
for regr_model in list_of_models:
    score_vector_rmse = utility.cross_validate_rmse(data_frame, N_repetitions , regr_model)
    mean_scores.append(np.mean(score_vector_rmse))
    std_scores.append(np.std(score_vector_rmse))

#plot C vs. accuracy
plt.fill_between(alpha_sweep , np.array(mean_scores) - np.array(std_scores) ,
                 np.array(mean_scores) + np.array(std_scores) , facecolor = 'xkcd:light pink', alpha=0.7)
plt.semilogx(alpha_sweep,mean_scores , color= 'xkcd:red' , linewidth=4)
# plt.semilogx(alpha_sweep,mean_scores , 'ok' , linewidth=4)
plt.xlabel('Hyper-parameter')
plt.ylabel('RMSE')
plt.grid(True, which='both')
plt.rc('grid', linestyle="--", color='grey', alpha=0.5)
plt.tight_layout()
plt.show()