# -*- coding: utf-8 -*-
import pandas as pd
ad_static = pd.read_table('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/ads_data/map_ad_static.out',header=None,sep='\t')
ad_static.columns= ['ad_id','creat_time','ad_account_id','Commodity_id','Commodity_type',
                    'Ad_Industry_Id','Ad_material_size']
print("ad_static:")
print(ad_static.head())

user = pd.read_table('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/user_data/user_data.out',nrows =100,header=None,sep='\t')
user.columns= ['user_id','age','gender','area','status',
               'education','consuptionability','device','work','ConnectionType','behavior']
print("user:")
print(user)

trace = pd.read_table('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/track_log/track_log_20190410.out',nrows =100,header=None,sep='\t')
trace.columns= ['ad_request_id','ad_request_time','user_id','ad_size_id','ad_list']
print("trace:")
print(trace)

trace_ad_list = trace[['ad_request_id','ad_list']]
print("trace_ad_list:")
print(trace_ad_list)



ad_operation = pd.read_table('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/ads_data/final_map_bid_opt.out',nrows =100,header=None,sep='\t')
ad_operation.columns= ['ad_id','creat_time','operation_type','transform_type',
                       'ad_type','ad_bid']
print("ad_operation:")
print(ad_operation)