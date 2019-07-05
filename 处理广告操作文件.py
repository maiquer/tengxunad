# -*- coding: utf-8 -*-

import pandas as pd
 
Exposure_Log_Data = pd.read_csv('/cos_person/tencent/train/Ad_Static_Feature_Data.csv')
 
Ad_id_in_static = Exposure_Log_Data['ad_id']
Ad_time_in_static = Exposure_Log_Data['Creation_time']

list_Ad_id_in_static = list(Ad_id_in_static)
list_Ad_time_in_static = list(Ad_time_in_static)
 
# 将广告操作对应的数据集(ad_operation.dat)进行清洗 清洗的内容包括一下几个部分
Ad_Operation_Data = []
# 定义操作数据对应的序列
 
All_kind_ad = []
with open('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/ads_data/final_map_bid_opt.out', 'r') as f:
    for i, line in enumerate(f):
        line = line.strip().split('\t')
        if (i % 10000) == 0:
            print("***********run%d"%(i))
        # 首先需要判断该条数据是否在静态数据集之中 不存在则删除
        if int(line[0]) not in list_Ad_id_in_static:
            # print("*******该条数据不存在于静态数据集之中，需要删除*******", line)
            continue
        # 首先需要广告操作数据集中的训练时间
        if len(line[1]) == 14:
            data_list = list(line[1])
            data_list.insert(4, '-')
            data_list.insert(7, '-')
            data_list.insert(10, ' ')
            data_list.insert(13, ':')
            data_list.insert(16, ':')
            line[1] = ''.join(data_list)
 
#        if line[2] == '1':
#            continue
        Ad_Operation_Data.append(line)

Ad_Operation_Data = pd.DataFrame(Ad_Operation_Data)
Ad_Operation_Data.columns = ['ad_id', 'Create_modify_time', 'Operation_type', 
                             'Transform_type', 'Charge_type','ad_bid']
Ad_Operation_Data.drop_duplicates(['ad_id'],keep='first', inplace=True)
Ad_Operation_Data.to_csv('/cos_person/tencent/train/Ad_Operation_Data.csv', index=False)