# -*- coding: utf-8 -*-

import pandas as pd

test_id = pd.read_csv('/cos_person/tencent/train/all_test.csv')
test_tmp = test_id[:100]

test_id = test_id[['sample_id','ad_id']]
test_id.to_csv('/cos_person/tencent/train/test_id.csv',index=None)

test = pd.read_csv('/cos_person/tencent/train/code_test.csv')
test_tmp = test[:10]

test = pd.concat( [test, test_id], axis=1)
test.drop(columns=['ad_bid'],inplace=True)
del test_id

test_basic = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_sample_bid.out',header=None,sep='\t')
test_basic.columns = ['sample_id','ad_id','Transform_type','Charge_type','ad_bid']
test_basic = test_basic[['ad_id','ad_bid']]


final_test = test_basic.merge(test, on=['ad_id'], how='left')

final_test.to_csv('/cos_person/tencent/train/all_test_with_bid.csv',index=None)

final_test_tmp = final_test[:10]
final_test = final_test[['Ad_pos_id', 'ad_bid', 'tfa_day', 'tfa_hour','tfa_minute', 
                         'Ad_account_id','Ad_Industry_Id', 'Ad_material_size',
                         'Commodity_type', 
                         'Transform_type', 'Charge_type', 
                         'Age', 'Gender','Area', 'Marriage_Status', 'Education', 'Consuption_Ability', 'Device',
                         'Work_Status', 'Connection_Type', 'Behavior']]
final_test.to_csv('/cos_person/tencent/train/test_with_ad_bid_request.csv',index=None)