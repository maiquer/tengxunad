# -*- coding: utf-8 -*-

import pandas as pd
trace = pd.read_table('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/track_log/track_log_20190410.out',nrows =100,header=None,sep='\t')
trace.columns= ['ad_request_id','ad_request_time','user_id','ad_size_id','ad_list']
print("trace:")
print(trace)

trace_ad_list = trace['ad_list']
print("trace_ad_list:")
print(trace_ad_list)

ad_list = []
ad_list_columns = ['ad_id', 'bid', 'pctr', 'quality_ecpm', 'totalEcpm', 
                       'filter', 'label']
ad_list.append(ad_list_columns)

for i in range(len(trace_ad_list)):
    
    tmp_line = trace_ad_list[i].strip().split(';')
 
    for each in tmp_line:
        save_line = []
        each_list = each.split(',')
        for i in range(7):
            save_line.append(each_list[i])
        ad_list.append(save_line)
    
ad_list = pd.DataFrame(ad_list)
print(ad_list)