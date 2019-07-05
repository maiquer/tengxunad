# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

train_x = pd.read_csv('/cos_person/tencent/train/code_train.csv')
train_y = pd.read_csv('/cos_person/tencent/train/train_label.csv')

x_test =  pd.read_csv('/cos_person/tencent/train/code_test.csv')

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2019)
y_pred_lgb = np.zeros(len(x_test))

params = {
           'objective': 'regression',
            'num_leaves': 5,
            'learning_rate': 0.05,
            'n_estimators': 720,
            'max_bin': 55,
            'bagging_fraction': 0.8,
            'bagging_freq': 5, 
            'feature_fraction': 0.2319,
            'feature_fraction_seed': 9, 
            'bagging_seed': 9,
            'min_data_in_leaf': 6,
            'min_sum_hessian_in_leaf': 11
        }

for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(train_x,train_y))):
    print('Fold', fold_n, 'started at', time.ctime())
    
    x_train, x_valid = train_x.iloc[train_index], train_x.iloc[valid_index]
    y_train, y_valid = train_y[train_index], train_y[valid_index]
     
    model = lgb.LGBMRegressor(**params, n_estimators = 2000, n_jobs = -1)
    model.fit(x_train, y_train, 
             eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_metric='mae',
             verbose=500, early_stopping_rounds=200)
            
    y_pred_valid = model.predict(x_valid)
    y_pred_lgb += model.predict(x_test, num_iteration=model.best_iteration_) / folds.n_splits

ad_request = pd.read_csv('/cos_person/tencent/train/test_ad_id_with_requeset_id.csv',header=None)
ad_request.columns = ['ad_id','Ad_Request_id','Ad_pos_id'] 
  
ad_request['ecpm'] = y_pred_lgb
ad_request.to_csv('/cos_person/tencent/train/ad_request_ecpm.csv',index=None)

request_ecpm = pd.read_csv('/cos_person/tencent/train/max_total.csv',header=None)
ad_request.merge(request_ecpm, on=['Ad_Request_id','Ad_pos_id'], how='left')

ad_request = ad_request[['ad_id','Ad_Request_id','Ad_pos_id','ecpm','test_totalEcpm']]

def update_ecpm(x,y):
    return max(x,y)
ad_request['ecpm_update'] = ad_request.apply(lambda row: update_ecpm(row['ecpm'], row['test_totalEcpm']), axis=1)    

temp = ad_request.groupby(['Ad_Request_id','Ad_pos_id'])['ecpm_update'].agg({'ecpm_max':'max'})

ad_request = ad_request.merge(temp, on=['Ad_Request_id','Ad_pos_id'], how='left')

def check_max_ecpm(x,y):
    if x == y:
        return 1
    else:
        return 0
ad_request['res_count'] = ad_request.apply(lambda row: check_max_ecpm(row['ecpm'], row['ecpm_max']), axis=1) 

temp = ad_request.groupby(['ad_id'])['res_count'].agg({'count_expose':'count'})

ad_request = ad_request.merge(temp, on=['ad_id'], how='left')

day422 = pd.read_csv('/cos_person/tencent/train/0422.csv',header=None)
day422.columns = ['ad_id','expose']

test_change = ad_request.merge(day422, on=['ad_id'], how='left')
test_change['expose'].fillna(test_change['expose'].mean(), inplace=True)

num = pd.read_csv('./train/历史信息.csv')
num.columns = ['sample_id','ad_id','base_num_click','find']

result_ronghe = res[[0,1,10,'num_click']]
result_ronghe.columns = ['sample_id','ad_id','ad_bid','num_click']

result_ronghe = result_ronghe.merge(num, on=['sample_id','ad_id'], how='left')

result_ronghe['sub'] = result_ronghe['base_num_click'] * result_ronghe['find']\
                     + result_ronghe['num_click'] * (1-result_ronghe['find'])
                     
result_ronghe['res'] =  result_ronghe['sub'].values + result_ronghe['ad_bid'].values/1000                     

submission = result_ronghe[['sample_id','res']]
submission.to_csv('./res/lgb_submission.csv',index=None,header=None) 

###############################################################################

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2019)
y_pred_lgb = np.zeros(len(X_test))

for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(x,y))):
    print('Fold', fold_n, 'started at', time.ctime())
    
    x_train, x_valid = x.iloc[train_index], x.iloc[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    catboost_model = CatBoostRegressor(
#                                       objective = smape_objective,
                                       custom_metric= 'SMAPE',
                                       eval_metric = 'SMAPE',
#                                       learning_rate = 0.5,
#                                       l2_leaf_reg = 10,
#                                       depth = 6,
#                                       subsample = 0.8,
                                       early_stopping_rounds = 5,
                                       num_trees = 200,
                                       loss_function='MAE',
                                       verbose=True)   
    train_pool = Pool(x_train, y_train)
    val_pool = Pool(x_valid, y_valid)
    catboost_model.fit(train_pool, eval_set=val_pool,verbose_eval=100)

    test_pool = Pool(X_test)
    y_pred_lgb += catboost_model.predict(test_pool) / folds.n_splits
    
res = pd.read_csv('./train/update_Btest_sample.dat', header=None,sep='\t')
res['num_click'] = y_pred_lgb

num = pd.read_csv('./train/历史信息.csv')
num.columns = ['sample_id','ad_id','base_num_click','find']

result_ronghe = res[[0,1,10,'num_click']]
result_ronghe.columns = ['sample_id','ad_id','ad_bid','num_click']

result_ronghe = result_ronghe.merge(num, on=['sample_id','ad_id'], how='left')

result_ronghe['sub'] = result_ronghe['base_num_click'] * result_ronghe['find']\
                     + result_ronghe['num_click'] * (1-result_ronghe['find'])
                     
result_ronghe['res'] =  result_ronghe['sub'].values + result_ronghe['ad_bid'].values/1000                     

submission = result_ronghe[['sample_id','res']]
submission.to_csv('./res/cat_submission.csv',index=None,header=None) 

########################################################################

from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

def smape_error(preds, train_data):
	labels = train_data.get_label()
	return 'error', 100 * np.mean(np.fabs(preds - labels) / (preds + labels) * 2)


folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2019)
y_pred_lgb = np.zeros(len(X_test))

params = {
            'colsample_bytree': 0.4603,
            'gamma': 0.0468,
            'learning_rate': 0.05, 
            'max_depth': 3,
            'min_child_weight': 1.7817,
            'n_estimators': 2200,
            'reg_alpha': 0.4640, 
            'reg_lambda': 0.8571,
            'subsample': 0.5213,
            'silent': 1,
            'random_state': 2019, 
}

for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(x,y))):
    print('Fold', fold_n, 'started at', time.ctime())
    
    x_train, x_valid = x.iloc[train_index], x.iloc[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    xgb_train = xgb.DMatrix(x_train, y_train)
    xgb_val = xgb.DMatrix(x_valid, y_valid)

    model = xgb.train(params, xgb_train, num_boost_round=200,
                      evals=[(xgb_train,'train'),(xgb_val,'val')], 
                      obj = smape_objective, 
                      feval=smape_error,
                      verbose_eval=100, 
                      early_stopping_rounds=100)
       
    y_pred_valid = model.predict(xgb_val)
    
    xgb_test = xgb.DMatrix(X_test)
    y_pred_lgb += model.predict(xgb_test, ntree_limit=model.best_ntree_limit) / folds.n_splits
    
res = pd.read_csv('./train/update_Btest_sample.dat', header=None,sep='\t')
res['num_click'] = y_pred_lgb

num = pd.read_csv('./train/历史信息.csv')
num.columns = ['sample_id','ad_id','base_num_click','find']

result_ronghe = res[[0,1,10,'num_click']]
result_ronghe.columns = ['sample_id','ad_id','ad_bid','num_click']

result_ronghe = result_ronghe.merge(num, on=['sample_id','ad_id'], how='left')

result_ronghe['sub'] = result_ronghe['base_num_click'] * result_ronghe['find']\
                     + result_ronghe['num_click'] * (1-result_ronghe['find'])
                     
result_ronghe['res'] =  result_ronghe['sub'].values + result_ronghe['ad_bid'].values/1000                     

submission = result_ronghe[['sample_id','res']]
submission.to_csv('./res/xg_submission.csv',index=None,header=None) 