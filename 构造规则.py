# -*- coding: utf-8 -*-

import pandas as pd

test = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_sample_bid.out',header=None,sep='\t')
test.columns = ['sample_id','ad_id','transform_type','charge_type','bid']

test_list = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/final_select_test_request.out',header=None,sep='\t')
test_list.columns = ['ad_id','request_list']

test_log = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_tracklog_20190423.last.out',header=None,sep='\t')
test_log.columns = ['Ad_Request_id', 'Ad_Request_Time','user_id','Ad_pos_id', 'ad_list']

test_log['max_total'] = 0.0

for i, line in enumerate(test_log['ad_list']):
    tmp_line = line.strip().split(';')
    max_total = 0.0         
    for each in tmp_line:
        each_list = each.split(',')
        if each_list[5] == '1':
            continue
        else: 
            if float(each_list[4])>max_total:
                max_total = float(each_list[4])
    test_log['max_total'][i] = max_total 
    
res = test_log[['Ad_Request_id','Ad_pos_id','max_total']]   
res.to_csv('/cos_person/tencent/train/max_total.csv',index=None,header=None)  

test_simple = test[['sample_id','ad_id']]
test_simple = test_simple.merge(test_list, on=['ad_id'], how='left')

res['index_id'] = res['Ad_Request_id'].astype(str)+','+ res['Ad_pos_id'].astype(str)
requeset_id_list = list(res['index_id'])

target = []
for i, line in enumerate(test_simple['request_list']):
    tmp_line = line.strip().split('|')
    
    save_line = [] 

    for each in tmp_line: 
        temp_line = [] 
        if each not in requeset_id_list:
            save_line.append (999999)
        else:
            temp_line.append((int(each_list[0]),int(each_list[1])))
            temp_line = pd.DataFrame(temp_line)
            temp_line.columns = ['Ad_Request_id','Ad_pos_id']
            temp_line = temp_line.merge(res,on=['Ad_Request_id','Ad_pos_id'], how='left')
            save_line.append (temp_line['max_total'].values)         
    target.append(save_line)
target = pd.DataFrame(target)
    
test_simple['target_epcm'] =  target[0]

test_change = test_simple[['sample_id','ad_id','target_epcm']]

day422 = pd.read_csv('/cos_person/tencent/train/0422.csv',header=None)
day422.columns = ['ad_id','expose']

test_change = test_change.merge(day422, on=['ad_id'], how='left')
test_change['expose'].fillna(0, inplace=True)

test_change.to_csv('/cos_person/tencent/train/test_change.csv',index=None,header=None)

test_change['count'] = 0

for i, line in enumerate(test_change['target_epcm']):
    count = 0
    tmp_line = line.strip().split(',')
      
    for each in tmp_line:
        if test_change['expose'][i] > float(each):
            count = count+1
    test_change['count'][i] = count

result = test_change[['sample_id','count']]
test_bid = test[['sample_id','bid']]
result = result.merge(test_bid, on=['sample_id'], how='left')

result['res'] =  result['count'].values + result['bid'].values/10000
submission = result[['sample_id','res']]
submission.to_csv('/cos_person/tencent/res/submission.csv',index=None,header=None)   