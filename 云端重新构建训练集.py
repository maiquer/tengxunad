# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import numpy as np

expose_train = pd.read_csv('/cos_person/tencent/train/Train_Log_Data.csv',header=0)

expose_train.columns = ['Ad_Request_id', 'Ad_Request_Time',
                        'user_id','Ad_pos_id', 
                        'ad_id','ad_bid','pctr',
                        'quality_ecpm', 'totalEcpm',
                        'filter', 'label']

f=lambda x:datetime.datetime.utcfromtimestamp(x+8*3600).strftime("%Y-%m-%d %H:%M:%S")
    
expose_train['time'] = expose_train['Ad_Request_Time'].apply(f)

tfa = expose_train.time.astype(str).apply(lambda x: datetime.datetime(
                                          int(x[:4]),
                                          int(x[5:7]),
                                          int(x[8:10]),
                                          int(x[11:13]),
                                          int(x[14:16]),
                                          int(x[17:])))
 
expose_train['tfa_day'] = np.array([x.day for x in tfa])

expose_train_show = expose_train[:10]
#保留时间（后序展开队列使用）
expose_train.drop(['filter', 'label'], axis=1,inplace=True)

Ad_Static_Data = pd.read_csv('/cos_person/tencent/train/Ad_Static_Feature_Data.csv')
#Ad_Static_Data.drop('Commodity_id', axis=1, inplace=True) ##这里我认为商品id是没有参考价值的
#Ad_Static_Data.drop('Ad_account_id', axis=1, inplace=True) ##这里我认为广告账户id是具有参考价值的
Ad_Static_Data.drop('Creation_time', axis=1, inplace=True) ##这里我认为广告创建时间没办法使用
print("*********static:\n", Ad_Static_Data.info())

Merce_Ad_Static_and_Exposure_Data = pd.merge(expose_train, Ad_Static_Data, on=['ad_id'])
 
# 读取广告操作数据集并拼接数据集
Op_Ad_Data = pd.read_csv('/cos_person/tencent/train/Ad_Operation_Data.csv').drop_duplicates(['ad_id'])
Op_Ad_Data.drop('Create_modify_time', axis=1, inplace=True)
Op_Ad_Data.drop('Operation_type', axis=1, inplace=True)
Op_Ad_Data.drop('ad_bid', axis=1, inplace=True)
 
Dataset_For_Train = pd.merge(Merce_Ad_Static_and_Exposure_Data,Op_Ad_Data, on=['ad_id'])
print("*************ad done!\n", Dataset_For_Train.info())

user_Data = pd.read_csv('/cos_person/tencent/train/userFeature.csv').drop_duplicates(['user_id'])

result = pd.merge(Dataset_For_Train,user_Data, on=['user_id'])
##为后序队列展开做准备
result.to_csv('/cos_person/tencent/train/Dataset_For_Train.csv', index=False)



