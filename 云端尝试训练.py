# -*- coding: utf-8 -*-
import os as input
input.system('pip install tqdm')
input.system('pip install lightgbm')
input.system('pip install catboost')

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold

['Ad_pos_id', 'ad_bid', 'tfa_day', 'tfa_hour', 'tfa_minute',
 'Ad_account_id', 'Ad_Industry_Id', 'Ad_material_size', 'Commodity_type',
 'Transform_type', 'Charge_type', 'Age', 'Gender', 'Area',
 'Marriage_Status', 'Education', 'Consuption_Ability', 'Device',
 'Work_Status', 'Connection_Type', 'Behavior', 'istest']

train_x = pd.read_csv('/cos_person/tencent/train/code_train.csv')
train_x.drop(columns=['ad_bid','istest'],inplace=True)

train_x_tmp = train_x[:10]

train_y = pd.read_csv('/cos_person/tencent/train/train_label.csv')

x_test =  pd.read_csv('/cos_person/tencent/train/code_test.csv')
x_test.drop(columns=['ad_bid','istest'],inplace=True)
x_test_tmp = x_test[:10]

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=2019)

y_pred_lgb = np.zeros(len(x_test))
 
catboost_model = CatBoostRegressor(
                                   custom_metric= 'MAE',
                                   eval_metric = 'MAE',
                                   learning_rate = 0.1,
                                   l2_leaf_reg = 5,

                                   early_stopping_rounds = 100,
                                   num_trees = 2000,
                                   loss_function='MAE',
                                   verbose=True)   
train_pool = Pool(x_train, y_train)
val_pool = Pool(x_valid, y_valid)
catboost_model.fit(train_pool, eval_set=val_pool,verbose_eval=100)

test_pool = Pool(x_test)
y_pred_lgb = catboost_model.predict(test_pool)
 
result = pd.read_csv('/cos_person/tencent/train/test_id.csv')
['sample_id', 'ad_id']
   
result['ecpm'] = y_pred_lgb
result_tmp = result[:10]
request_ecpm = pd.read_csv('/cos_person/tencent/train/max_total.csv',header=None)
request_ecpm.columns = ['Ad_Request_id', 'Ad_Request_Time','user_id','Ad_pos_id',
                        'test_ad_id','test_ad_bid',
                        'test_pctr','test_quality_ecpm', 
                        'test_totalEcpm','test_filter']
request_ecpm_tmp = request_ecpm[:10]

result = pd.concat( [result, request_ecpm], axis=1)

result = result[['sample_id','ad_id','Ad_Request_id','Ad_pos_id','ecpm','test_totalEcpm']]

def update_ecpm(x,y):
    return max(x,y)
result['ecpm_update'] = result.apply(lambda row: update_ecpm(row['ecpm'], row['test_totalEcpm']), axis=1)    

temp = result.groupby(['Ad_Request_id','Ad_pos_id'])['ecpm_update'].agg({'ecpm_max':'max'}).reset_index()

result = result.merge(temp, on=['Ad_Request_id','Ad_pos_id'], how='left')
result_tmp = result[:10000]
def check_max_ecpm(x,y):
    if abs(x - y)<0.000001:
        return 1
    else:
        return 0
result['res_count'] = result.apply(lambda row: check_max_ecpm(row['ecpm'], row['ecpm_max']), axis=1) 

temp = result.groupby(['ad_id'])['res_count'].agg({'count_expose':'count'}).reset_index()

test_ad_basic = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_sample_bid.out',header=None,sep='\t')
test_ad_basic.columns = ['sample_id','ad_id','Transform_type','Charge_type','ad_bid']


test_ad = test_ad_basic.merge(temp, on=['ad_id'], how='left')
test_ad['res'] =  test_ad['count_expose'].values/100 + test_ad['ad_bid'].values/10000
test_ad.to_csv('/cos_person/tencent/res/net_result.csv',index=None)

sub = test_ad[['sample_id','res']]
sub.to_csv('/cos_person/tencent/res/submission.csv',index=None,header=None)
sub.describe()
print("done!")