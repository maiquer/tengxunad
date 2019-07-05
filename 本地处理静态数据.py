# -*- coding: utf-8 -*-

import pandas as pd
import time
 
Ad_Static_Feature_Data = []

# 定义相关列
Ad_Static_Feature_Data_columns = ['ad_id', 'Creation_time', 'Ad_account_id', 
                                  'Commodity_id', 'Commodity_type',
                                  'Ad_Industry_Id', 'Ad_material_size']
 
# 为数据集增加列名称
Ad_Static_Feature_Data.append(Ad_Static_Feature_Data_columns)
with open('map_ad_static.out', 'r') as f:
    for i, line in enumerate(f):
        line = line.strip().split('\t')
        # 分别用于判断该条广告数据是否存在记录缺失 是否创建时间为0 广告行业是否存在多值
        if line[1] == '0' :
            # print("数据集中创建时间为0的数据集是: ", line)
            continue
        
        if line[1] == '-1' :
            # print("数据集中创建时间为0的数据集是: ", line)
            continue
        
        if ',' in line[5]:
            # print("数据集中广告行业ID存在多值记录是: ", line)
            continue
        if len(line) != 7:
            # print("广告数据集中出现缺失数据: ", line)
            continue
        loacl_time = int(line[1])
        time_local = time.localtime(loacl_time)
        line[1] = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        Ad_Static_Feature_Data.append(line)

 
user_feature = pd.DataFrame(Ad_Static_Feature_Data)
user_feature.to_csv('./train/Ad_Static_Feature_Data.csv', index=False, header=False)
