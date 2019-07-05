# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime

['Ad_Request_id', 'Ad_Request_Time', 'user_id', 'Ad_pos_id', 'ad_id',
 'ad_bid', 'pctr', 'quality_ecpm', 'totalEcpm', 'tfa_day', 'tfa_hour',
 'tfa_minute', 'Ad_account_id', 'Commodity_type', 'Ad_Industry_Id',
 'Ad_material_size', 'Transform_type', 'Charge_type', 'Age', 'Gender',
 'Area', 'Marriage_Status', 'Education', 'Consuption_Ability', 'Device',
 'Work_Status', 'Connection_Type', 'Behavior']


['Ad_pos_id', 'ad_bid', 'totalEcpm', 'tfa_day', 'tfa_hour','tfa_minute', 
 'Ad_account_id', 'Commodity_type', 'Transform_type', 'Charge_type', 
 'Age', 'Gender','Area', 'Marriage_Status', 'Education', 'Consuption_Ability', 'Device',
 'Work_Status', 'Connection_Type', 'Behavior']

test_ad_basic = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_sample_bid.out',header=None,sep='\t')
test_ad_basic.columns = ['sample_id','ad_id','Transform_type','Charge_type','ad_bid']

test_ad_basic.drop_duplicates(['ad_id'],  keep='first', inplace=True) ##引入了bid的影响，需要后续考虑单调性

test_list = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/final_select_test_request.out',header=None,sep='\t')
test_list.columns = ['ad_id','request_list']

test_log = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_tracklog_20190423.last.out',header=None,sep='\t')
test_log.columns = ['Ad_Request_id', 'Ad_Request_Time','user_id','Ad_pos_id', 'ad_list']

test_log_df = pd.read_csv('/cos_person/tencent/train/max_total.csv',header=None)  
test_log_df.columns = ['Ad_Request_id', 'Ad_Request_Time','user_id',
                       'Ad_pos_id', 'test_ad_id','test_ad_bid',
                        'test_pctr','test_quality_ecpm', 
                        'test_totalEcpm','test_filter']

temp_line = pd.read_csv('/cos_person/tencent/train/test_ad_id_with_requeset_id.csv',header=None)       
temp_line.columns = ['ad_id','Ad_Request_id','Ad_pos_id']

temp_line = temp_line.merge(test_log_df, on=['Ad_Request_id','Ad_pos_id'], how='left')

test_request = temp_line.merge(test_ad_basic, on=['ad_id'], how='left')

userFeature = pd.read_csv('/cos_person/tencent/train/userFeature.csv').drop_duplicates(['user_id'])

test_user = test_request.merge(userFeature, on=['user_id'], how='left')

ad_static = pd.read_csv('/cos_person/tencent/train/Ad_Static_Feature_Data.csv',header=None)
ad_static.columns = ['ad_id', 'Creation_time', 'Ad_account_id', 
                     'Commodity_id', 'Commodity_type',
                     'Ad_Industry_Id', 'Ad_material_size']

final_test = test_user.merge(ad_static, on=['ad_id'], how='left')
final_test.to_csv('/cos_person/tencent/train/test_user.csv',index=None)

f=lambda x:datetime.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
    
final_test['time'] = final_test['Ad_Request_Time'].apply(f)

tfa = final_test.time.astype(str).apply(lambda x: datetime.datetime(
                                          int(x[:4]),
                                          int(x[5:7]),
                                          int(x[8:10]),
                                          int(x[11:13]),
                                          int(x[14:16]),
                                          int(x[17:])))
 
final_test['tfa_day'] = np.array([x.day for x in tfa])
final_test['tfa_hour'] = np.array([x.hour for x in tfa])
final_test['tfa_minute'] = np.array([x.minute for x in tfa])

test_basic = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_sample_bid.out',header=None,sep='\t')
test_basic.columns = ['sample_id','ad_id','Transform_type','Charge_type','ad_bid']
test_basic = test_basic[['ad_id','ad_bid']]

final_test = test_basic.merge(final_test, on=['ad_id'], how='left')

final_test = final_test[['Ad_pos_id', 'ad_bid', 'tfa_day', 'tfa_hour','tfa_minute', 
                         'Ad_account_id','Ad_Industry_Id', 'Ad_material_size',
                         'Commodity_type', 'Commodity_id',
                         'Transform_type', 'Charge_type', 
                         'Age', 'Gender','Area', 'Marriage_Status', 'Education', 'Consuption_Ability', 'Device',
                         'Work_Status', 'Connection_Type', 'Behavior']]
final_test.to_csv('/cos_person/tencent/train/test.csv',index=None)