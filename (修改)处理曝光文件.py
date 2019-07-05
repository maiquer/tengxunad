# -*- coding: utf-8 -*-

import pandas as pd
 
 
# 需要读取广告操作数据集中的广告ID并将其转化成list
Ad_operation = pd.read_csv('/cos_person/tencent/train/Ad_Operation_Data.csv')
Ad_op_id = Ad_operation['ad_id'].drop_duplicates(keep='first', inplace=False)
list_Ad_op_id = list(Ad_op_id)

user_feature = pd.read_csv('/cos_person/tencent/train/userFeature.csv')
user_id = user_feature['user_id'].drop_duplicates(keep='first', inplace=False)
list_user_id = list(user_id)
 
# 定义曝光日志中的相关列
Exposure_Log_Data = []

for j in range(10,23):
    with open('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/track_log/track_log_201904' + str(j) + '.out', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            
            flag_line = line
            
            if (i % 5000000) == 0:
                print("*******run ", i)
    
            if line[0] == '0' or line[1] == '0' or line[2] == '0' or line[3] == '0' or line[4] == '0':
                continue
       
            if ',' in line[2]:
                continue
            
            tmp_user_id = int(line[2]) ##不在用户特征集中的曝光
            if tmp_user_id not in list_user_id:
                continue
            
            if '.' in line[0]:
                continue
            if '.' in line[3]:
                continue
            
            ad_list = []
            ad_list_columns = ['ad_id', 'bid', 'pctr', 'quality_ecpm', 'totalEcpm', 
                                   'filter', 'label']
            ad_list.append(ad_list_columns)
            
            tmp_line = line[4].strip().split(';')
             
            for each in tmp_line:
                save_line = []
                each_list = each.split(',')
                if each_list[6] != '1':
                    continue
                else:
                    if int(each_list[0]) not in list_Ad_op_id:
                        continue;
                    else:  
                        line.append(int(each_list[0]))
                        line.append(int(each_list[1]))
                        line.append(float(each_list[2]))
                        line.append(float(each_list[3]))
                        line.append(float(each_list[4]))
                        line.append(int(each_list[5]))
                        line.append(int(each_list[6]))
                        save_line.append(line[5])
                if save_line:
                    Exposure_Log_Data.append(line)
                    line = flag_line
                    
Exposure_Log_Data = pd.DataFrame(Exposure_Log_Data) 
Exposure_Log_Data_columns = ['Ad_Request_id', 'Ad_Request_Time','user_id',
                             'Ad_pos_id', 'ad_list',
                             'ad_id','ad_bid','pctr',
                             'quality_ecpm', 'totalEcpm',
                             'filter', 'label']
Exposure_Log_Data.columns  = Exposure_Log_Data_columns  
Exposure_Log_Data.to_csv('/cos_person/tencent/train/Total_Exposure_Log_Data_with_AD_list.csv', index=False,header=None)

Exposure_Log_Data.drop(Exposure_Log_Data.columns[['ad_list']], axis=1,inplace=True)
Exposure_Log_Data.to_csv('/cos_person/tencent/train/Train_Log_Data.csv', index=False,header=None)