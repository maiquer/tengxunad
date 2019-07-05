# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

train = pd.read_csv('/cos_person/tencent/train/Dataset_For_Train.csv')
train_tmp = train[:10]
['Ad_Request_id', 'Ad_Request_Time', 'user_id', 'Ad_pos_id', 'ad_id',
 'ad_bid', 'pctr', 'quality_ecpm', 'totalEcpm', 'tfa_day', 'tfa_hour',
 'tfa_minute', 'Ad_account_id', 'Commodity_type', 'Ad_Industry_Id',
 'Ad_material_size', 'Transform_type', 'Charge_type', 'Age', 'Gender',
 'Area', 'Marriage_Status', 'Education', 'Consuption_Ability', 'Device',
 'Work_Status', 'Connection_Type', 'Behavior']

#暂时只考虑total
all_train = train[['Ad_pos_id', 'ad_bid', 'tfa_day', 'tfa_hour','tfa_minute', 
                   'Ad_account_id','Ad_Industry_Id', 'Ad_material_size',
                   'Commodity_type', 
                   'Transform_type', 'Charge_type', 
                   'Age', 'Gender','Area', 'Marriage_Status', 'Education', 'Consuption_Ability', 'Device',
                   'Work_Status', 'Connection_Type', 'Behavior',
                   'totalEcpm']]
label = train[['totalEcpm']]

all_train.drop(columns=['totalEcpm'],inplace=True)
all_train['istest'] = 0
all_train_tmp = all_train[:100]

test = pd.read_csv('/cos_person/tencent/train/all_test.csv')
test_tmp = test[:100]

test = test[['Ad_pos_id', 'ad_bid', 'tfa_day', 'tfa_hour','tfa_minute', 
                'Ad_account_id','Ad_Industry_Id', 'Ad_material_size',
                'Commodity_type', 
                'Transform_type', 'Charge_type', 
                'Age', 'Gender','Area', 'Marriage_Status', 'Education', 'Consuption_Ability', 'Device',
                'Work_Status', 'Connection_Type', 'Behavior']]

test['istest'] = 1

test = pd.concat( [all_train, test], axis=0)
test_tmp = test[:10]
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

for i in ["Ad_account_id","Ad_Industry_Id","Ad_material_size","Commodity_type",
          "Age","Gender","Area","Marriage_Status", "Education",
          "Consuption_Ability", "Device","Work_Status", "Connection_Type", "Behavior"]:
    test[i] = lbl.fit_transform(test[i].values)
    
train = test.loc[(test['istest'] == 0)]
train.to_csv('/cos_person/tencent/train/code_train.csv',index=None)

label.to_csv('/cos_person/tencent/train/train_label.csv',index=None)

test = test.loc[(test['istest'] == 1)]
test.to_csv('/cos_person/tencent/train/code_test.csv',index=None)