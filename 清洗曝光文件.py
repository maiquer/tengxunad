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
Exposure_Log_Data_columns = ['Ad_Request_id', 'Ad_Request_Time','user_id','Ad_pos_id', 'ad_list',
                             'ad_id','ad_bid','pctr','quality_ecpm', 'totalEcpm','filter', 'label']

Exposure_Log_Data.append(Exposure_Log_Data_columns)

for i in range(10,23):
    with open('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/track_log/track_log_201904' + str(i) + '.out', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
        
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
                if each_list[6] != 1:
                    continue
                else:
                    if each_list[0] not in list_Ad_op_id:
                        continue;
                    else:    
                        for i in range(7):
                            line[5+i] = each_list[i]
                            save_line.append(each_list[i])
                        break
            if not save_line:
                Exposure_Log_Data.append(line)
                
Exposure_Log_Data = pd.DataFrame(Exposure_Log_Data)             
Exposure_Log_Data.to_csv('/cos_person/tencent/train/Total_Exposure_Log_Data_with_AD_list.csv', index=False,header=None)

Train_Log_Data = Exposure_Log_Data.drop('ad_list', axis=1);
Train_Log_Data.to_csv('/cos_person/tencent/train/Train_Log_Data.csv', index=False)