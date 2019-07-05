# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv('./train/train_hebing.csv',header = 0)

y = X_train.num_click.values

X_train = X_train[['Ad_material_size', 'Ad_Industry_Id', 'Commodity_type',
                  'Commodity_id','Ad_account_id','Delivery_time',
                  
                  'date','week',
                  
                  'last1_num_click','last2_num_click','last3_num_click',
                  
                  'Delivery_time_mean',
                  'Ad_account_id_mean',
                  'Commodity_id_mean',
                  'Commodity_type_mean',
                  'Ad_Industry_Id_mean',
                  'Ad_material_size_mean',
                  'Ad_material_size&Ad_Industry_Id','Ad_material_size&Ad_Industry_Id_mean',
                  'material_size&Commodity_type','material_size&Commodity_type_mean',
                  'material_size&Commodity_id','material_size&Commodity_id_mean',
                  'material_size&Ad_account_id','material_size&Ad_account_id_mean',
                  'material_size&Delivery_time','material_size&Delivery_time_mean',
                  'Ad_Industry_Id&Commodity_type','Ad_Industry_Id&Commodity_type_mean',
                  'Ad_Industry_Id&Commodity_id','Ad_Industry_Id&Commodity_id_mean',
                  'Ad_Industry_Id&Ad_account_id','Ad_Industry_Id&Ad_account_id_mean',
                  'Ad_Industry_Id&Delivery_time','Ad_Industry_Id&Delivery_time_mean',
                  'Commodity_type&industry_id','Commodity_type&industry_id_mean',
                  'Commodity_type&Commodity_id','Commodity_type&Commodity_id_mean',
                  'Commodity_type&Ad_account_id','Commodity_type&Ad_account_id_mean',
                  'Commodity_type&Delivery_time','Commodity_type&Delivery_time_mean',
                  'Commodity_id&Ad_account_id','Commodity_id&Ad_account_id_mean',
                  'Commodity_id&Delivery_time','Commodity_id&Delivery_time_mean',
                  'Ad_account_id&Delivery_time','Ad_account_id&Delivery_time_mean']]
x = X_train

x_test =  pd.read_csv('./train/test_hebing.csv',header = 0)
X_test = x_test[['Ad_material_size', 'Ad_Industry_Id', 'Commodity_type',
                  'Commodity_id','Ad_account_id','Delivery_time',
                  
                  'date','week',
                  
                  'last1_num_click','last2_num_click','last3_num_click',
                  
                  'Delivery_time_mean',
                  'Ad_account_id_mean',
                  'Commodity_id_mean',
                  'Commodity_type_mean',
                  'Ad_Industry_Id_mean',
                  'Ad_material_size_mean',
                  'Ad_material_size&Ad_Industry_Id','Ad_material_size&Ad_Industry_Id_mean',
                  'material_size&Commodity_type','material_size&Commodity_type_mean',
                  'material_size&Commodity_id','material_size&Commodity_id_mean',
                  'material_size&Ad_account_id','material_size&Ad_account_id_mean',
                  'material_size&Delivery_time','material_size&Delivery_time_mean',
                  'Ad_Industry_Id&Commodity_type','Ad_Industry_Id&Commodity_type_mean',
                  'Ad_Industry_Id&Commodity_id','Ad_Industry_Id&Commodity_id_mean',
                  'Ad_Industry_Id&Ad_account_id','Ad_Industry_Id&Ad_account_id_mean',
                  'Ad_Industry_Id&Delivery_time','Ad_Industry_Id&Delivery_time_mean',
                  'Commodity_type&industry_id','Commodity_type&industry_id_mean',
                  'Commodity_type&Commodity_id','Commodity_type&Commodity_id_mean',
                  'Commodity_type&Ad_account_id','Commodity_type&Ad_account_id_mean',
                  'Commodity_type&Delivery_time','Commodity_type&Delivery_time_mean',
                  'Commodity_id&Ad_account_id','Commodity_id&Ad_account_id_mean',
                  'Commodity_id&Delivery_time','Commodity_id&Delivery_time_mean',
                  'Ad_account_id&Delivery_time','Ad_account_id&Delivery_time_mean']]

def fgrad(y_pred,y_true):
    return (2*y_true*(y_pred-y_true))/ (np.square(y_pred + y_true) * abs(y_pred - y_true))

def fhess(y_pred,y_true):
    return (-4 *y_true*(y_pred -y_true))/(pow((y_pred + y_true),3) * abs(y_pred - y_true))


def smape_objective(preds,train_data):
    labels = train_data.get_label()
    grad = fgrad(preds,labels)
    hess = fhess(preds,labels)
    return grad, hess


def smape_error(preds, train_data):
	labels = train_data.get_label()
	return 'error', 100*np.mean(np.fabs(preds - labels) / (preds + labels) * 2), False

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2019)
y_pred_lgb = np.zeros(len(X_test))

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

for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(x,y))):
    print('Fold', fold_n, 'started at', time.ctime())
    
    x_train, x_valid = x.iloc[train_index], x.iloc[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
     
    train_lgb = lgb.Dataset(x_train,y_train)
    val_lgb = lgb.Dataset(x_valid,y_valid)
    
    lgbm = lgb.train(params,
                train_lgb,
#                num_boost_round  = 1000,
                valid_sets = [val_lgb,train_lgb],
                early_stopping_rounds = 200,
                fobj = smape_objective,
                feval  = smape_error,
                verbose_eval = 100 
               )
            
    y_pred_lgb += lgbm.predict(X_test, num_iteration=lgbm.best_iteration) / folds.n_splits
    
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