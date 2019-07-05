# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

test = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_sample_bid.out',header=None,sep='\t')
test.columns = ['sample_id','ad_id','transform_type','charge_type','bid']

test_list = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/final_select_test_request.out',header=None,sep='\t')
test_list.columns = ['ad_id','request_list']

test_log = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_tracklog_20190423.last.out',header=None,sep='\t')
test_log.columns = ['Ad_Request_id', 'Ad_Request_Time','user_id','Ad_pos_id', 'ad_list']

test_log_temp = test_log[:10]

test_log_df = pd.read_csv('/cos_person/tencent/train/max_total.csv',header=None)  
test_log_df.columns = ['Ad_Request_id', 'Ad_Request_Time','user_id',
                       'Ad_pos_id', 'test_ad_id','test_ad_bid',
                        'test_pctr','test_quality_ecpm', 
                        'test_totalEcpm','test_filter']

test_simple = test[['sample_id','ad_id']]
test_simple = test_simple.merge(test_list, on=['ad_id'], how='left')

test_log_df['index_id'] = test_log_df['Ad_Request_id'].astype(str)+','+ test_log_df['Ad_pos_id'].astype(str)
test_log_df_show = test_log_df[:10]
requeset_id_list = list(set(test_log_df['index_id']))

temp_line = [] 

def func(x,y):
    line = str(x).strip().split('|')
    
    for each in line: 
        save_line = []
        save_line.append(y)
        each_list = each.split(',')
        
        save_line.append(int(each_list[0]))
        save_line.append(int(each_list[1]))
        
        temp_line.append(save_line)       
test_simple.drop_duplicates(['ad_id'],  keep='first', inplace=True) 
test_simple.apply(lambda row: func(row['request_list'], row['ad_id']), axis=1)

temp_line = pd.DataFrame(temp_line)
temp_line.to_csv('/cos_person/tencent/train/test_ad_id_with_requeset_id.csv',index=None,header=None)       

temp_line.columns = ['ad_id','Ad_Request_id','Ad_pos_id']
temp_line = temp_line.merge(test_log_df, on=['Ad_Request_id','Ad_pos_id'], how='left')

total = temp_line.groupby(['ad_id'])['test_totalEcpm'].apply(list).reset_index()
total.to_csv('/cos_person/tencent/train/test_total_totalEcpm.csv',index=None)
total_show = total[:2]

day422 = pd.read_csv('/cos_person/tencent/train/0422.csv',header=None)
day422.columns = ['ad_id','expose']

test_change = total.merge(day422, on=['ad_id'], how='left')
test_change['expose'].fillna(test_change['expose'].mean(), inplace=True)

test_change_show = test_change[:2]

def count(x,y):
    str_x = ",".join([str(i) for i in x])
    count = 0
    line = str_x.strip().split(',')
    
    for each in line: 
        if y>float(each):
            count = count+1
    return int(count) 
 
test_change['count'] = test_change.apply(lambda row: count(row['test_totalEcpm'], row['expose']), axis=1)

result = test_change[['ad_id','count']]
test_bid = test[['sample_id','ad_id','bid']]

sub = test_bid.merge(result, on=['ad_id'], how='left')

def min_sub(x):
    if x > 884.0:
        x = 884.0
    x = x/20
    return float(x)
sub['count'] = sub['count'].apply(min_sub)

sub['res'] =  sub['count'].values + sub['bid'].values/10000
submission = sub[['sample_id','res']]
submission.to_csv('/cos_person/tencent/res/submission.csv',index=None,header=None)   

submission.describe()
#
#tijiao = pd.read_csv('./res/529.csv',header=None)
#tijiao.describe()
#
#user = pd.read_csv('./train/userFeature.csv')
#use_id = list(set(user['user_id']))
#
#user_count = 0
#def count_userid(x):
#    global user_count
#    if x in use_id:
#        user_count = user_count + 1
#        
#test_log_df['user_id'].apply(count_userid)
#test_func_test_simple = test_simple[:1]
#test_func_test_simple.apply(lambda row: func(row['request_list'], row['ad_id']), axis=1)

#test_func_test_simple['request_list','ad_id'].apply(func)

#test_func_test_simple['request_list_res'] = test_func_test_simple['request_list'].apply(func)

#test_func_test_simple = test_simple[:1]
#test_func_test_simple.set_index(['ad_id']).request_list.str.split('|').apply(pd.Series).stack().reset_index(level=1,drop=True).to_frame('res')
#test_func_test_simple_show = test_func_test_simple.set_index(['ad_id']).request_list.str.split('|').apply(pd.Series).stack().reset_index(level=1,drop=True).to_frame('res')
#test_func_test_simple['request_list_res'] = test_func_test_simple['request_list'].apply(func)