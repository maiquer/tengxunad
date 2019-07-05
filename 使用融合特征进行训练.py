# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:28:09 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

X_train =  pd.read_csv('./train/data_statis.csv',header = 0)

y = X_train.num_click.values

X_train = X_train[['Ad_material_size', 'Ad_Industry_Id', 'Commodity_type',
                  'Commodity_id','Ad_account_id','Delivery_time','Delivery_time_mean',
                  'Ad_account_id_mean','Commodity_id_mean','Commodity_type_mean',
                  'Ad_Industry_Id_mean','Ad_material_size_mean']]
x = X_train

x_test =  pd.read_csv('./train/test_statis.csv',header = 0)
X_test = x_test[['Ad_material_size', 'Ad_Industry_Id', 'Commodity_type',
                  'Commodity_id','Ad_account_id','Delivery_time','Delivery_time_mean',
                  'Ad_account_id_mean','Commodity_id_mean','Commodity_type_mean',
                  'Ad_Industry_Id_mean','Ad_material_size_mean']]

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_lgb = np.zeros(len(X_test))

params = {'objective' : "regression", 
               'boosting':"gbdt",
               'metric':"mae",
               'boost_from_average':"false",
               'num_threads':8,
               'learning_rate' : 0.001,
               'num_leaves' : 52,
               'max_depth':-1,
               'tree_learner' : "serial",
               'feature_fraction' : 0.85,
               'bagging_freq' : 1,
               'bagging_fraction' : 0.85,
               'min_data_in_leaf' : 10,
               'min_sum_hessian_in_leaf' : 10.0,
               'verbosity' : -1}

for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(x,y))):
    print('Fold', fold_n, 'started at', time.ctime())
    
    x_train, x_valid = x.iloc[train_index], x.iloc[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
        
    model = lgb.LGBMRegressor(**params, n_estimators = 22000, n_jobs = -1)
    model.fit(x_train, y_train, 
                    eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_metric='mae',
                    verbose=1000, early_stopping_rounds=200)
            
    y_pred_valid = model.predict(x_valid)
    y_pred_lgb += model.predict(X_test, num_iteration=model.best_iteration_) / folds.n_splits
    
res = pd.read_csv('Btest_sample_new.dat', header=None,sep='\t')
res['num_click'] = y_pred_lgb
result = res[[0,1,10,'num_click']]
result.to_csv('submission_lgb.csv')

result['num_click'] =  result['num_click'].values + result[10].values/1000
sub = result[[0,'num_click']]  
sub.to_csv('./res/net_submission.csv',index=None,header=None)   

###################################################################################3
#num = pd.read_csv('./train/19_num_id.csv', header=None)
data = pd.read_csv('./train/Dataset_For_Train.csv')
tmp = res[[0,1,10,'num_click']]
id_list = list(set(data['ad_id']))

for i in range(len(tmp[1])):
    if tmp[1][i] in id_list :
        tmp['num_click'][i] = data.groupby(['ad_id'])['num_click'].mean()[tmp[1][i]]

tmp['num_click'] =  tmp['num_click'].values + tmp[10].values/1000
sub = tmp[[0,'num_click']]  
sub.to_csv('./res/guize_submission.csv',index=None,header=None)  














