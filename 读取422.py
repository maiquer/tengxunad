# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:53:18 2019

@author: Mr
"""
##读取422
import pandas as pd

expose = pd.read_csv('./track_log/track_log_20190422.out',header=None,sep='\t')
res = expose.loc[(expose[0] == 981495)]

test = pd.read_csv('test_sample_bid.out',header=None,sep='\t')

Exposure_Log_Data = []
Exposure_Log_Data_columns = ['request_id','ad_id','ad_bid','pctr','quality_ecpm', 
                             'totalEcpm','filter', 'label']

Exposure_Log_Data.append(Exposure_Log_Data_columns)

for i, line in enumerate(expose[4]):
    tmp_line = line.strip().split(';')
             
    for each in tmp_line:
        save_line = []
        each_list = each.split(',')
        if each_list[6] != '1':
            continue
        else: 
            save_line.append(int(expose[0][i]))
            save_line.append(int(each_list[0]))
            save_line.append(int(each_list[1]))
            save_line.append(float(each_list[2]))
            save_line.append(float(each_list[3]))
            save_line.append(float(each_list[4]))
            save_line.append(int(each_list[5]))
            save_line.append(int(each_list[6]))
            Exposure_Log_Data.append(save_line)
            
Exposure_Log_Data = pd.DataFrame(Exposure_Log_Data)
Exposure_Log_Data.drop(index=[0],inplace=True)
Exposure_Log_Data.columns = Exposure_Log_Data_columns

Exposure_Log_Data_temp = Exposure_Log_Data[:100]
Exposure_Log_Data_temp.drop(index=[0],inplace=True)

Exposure_Log_Data['totalEcpm'] = Exposure_Log_Data['totalEcpm'].astype(float)
tmp = Exposure_Log_Data.groupby(['ad_id'], as_index=False)['totalEcpm'].agg({'totalEcpm_mean':'mean'})
tmp.to_csv('./train/0422.csv',index=None,header=None)

