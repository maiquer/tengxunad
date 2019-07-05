# -*- coding: utf-8 -*-

import pandas as pd

test = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_sample_bid.out',header=None,sep='\t')
test.columns = ['sample_id','ad_id','transform_type','charge_type','bid']

test_list = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/final_select_test_request.out',header=None,sep='\t')
test_list.columns = ['ad_id','request_list']

test_log = pd.read_csv('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/test_data/test_tracklog_20190423.last.out',header=None,sep='\t')
test_log.columns = ['Ad_Request_id', 'Ad_Request_Time','user_id','Ad_pos_id', 'ad_list']

test_log_temp = test_log[:10]

max_total = []
for i, line in enumerate(test_log['ad_list']):
    tmp_line = line.strip().split(';')
    save_line = []
    max_num = 0.0 
    for each in tmp_line:
        
        each_list = each.split(',')
        if each_list[5] == '1':
            continue
        else: 
            if float(each_list[4])>max_num:
                max_num = float(each_list[4])
                save_line = []
                save_line.append(int(each_list[0]))
                save_line.append(int(each_list[1]))
                save_line.append(float(each_list[2]))
                save_line.append(float(each_list[3]))
                save_line.append(float(each_list[4]))
                save_line.append(int(each_list[5]))
    max_total.append(save_line)
max_total_df = pd.DataFrame(max_total)
max_total_df.columns = ['test_ad_id','test_ad_bid',
                        'test_pctr','test_quality_ecpm', 
                        'test_totalEcpm','test_filter']    

test_log_df = pd.concat( [test_log, max_total_df], axis=1) 
test_log_df.drop(columns=['ad_list'],inplace=True)   
test_log_df.to_csv('/cos_person/tencent/train/max_total.csv',index=None,header=None)  