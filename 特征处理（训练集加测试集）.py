# -*- coding: utf-8 -*-

import pandas as pd
train = pd.read_csv('./train/Dataset_For_Train.csv')
train['istest'] = 0
train = train[['ad_id','ad_bid','Delivery_time','Ad_account_id', 
            'Commodity_id','Commodity_type','Ad_Industry_Id','Ad_material_size',
           'tfa_month','tfa_day','num_click','istest']]

test = pd.read_csv('./train/update_Btest_sample.dat',header = None, sep='\t')

test.columns =  ['sample_id','ad_id','creat_time','Ad_material_size','Ad_Industry_Id',
                'Commodity_type','Commodity_id','Ad_account_id','Delivery_time','Chose_people','ad_bid']
 
test['tfa_month'] = 3
test['tfa_day'] = 20
test['num_click'] = 0
test['istest'] = 1

test = test[[   'ad_bid','ad_id','Delivery_time','Ad_account_id',
                'Commodity_id','Commodity_type','Ad_Industry_Id','Ad_material_size',
                'tfa_month','tfa_day','num_click','istest'
                ]]

all_data = pd.concat( [train, test], axis=0)

f = lambda x: x.split(',')[0]
all_data['Delivery_time'] = all_data['Delivery_time'].apply(f).astype('int64')

all_data['date'] = (all_data['tfa_month']-2)*28 + (all_data['tfa_day']-16)

all_data['week'] = (all_data['date'] + 6) % 7 

all_data['last1_date'] = all_data['date']-1
all_data.loc[all_data[all_data['last1_date']<0 ].index,['last1_date']]=0

all_data['last2_date'] = all_data['date']-2
all_data.loc[all_data[all_data['last2_date']<0 ].index,['last2_date']]=0

all_data['last3_date'] = all_data['date']-3
all_data.loc[all_data[all_data['last3_date']<0 ].index,['last3_date']]=0

################################################################
def get_refer_day(d):
    return d + 1
    
tmp = all_data.copy()   #  (192586,12)
tmp = tmp[['ad_id','num_click','date']]

tmp_df = tmp[tmp.date==0]  # 第一天的数据(8941,12)
tmp_df['date'] = tmp_df['date'] - 1   # 将第一天改为第0天
tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True) #(201527,12)
tmp['date'] = tmp['date'].apply(get_refer_day)    # 29,2,3,

tmp.columns = ['ad_id','last1_num_click','date']   

all_data = all_data.merge(tmp, on=['ad_id','date'], how='left')

################################################################
def get_refer_day(d):
    return d + 2
    
tmp = all_data.copy()   #  (192586,12)
tmp = tmp[['ad_id','num_click','date']]

tmp_df = tmp[tmp.date<=1]  # 第一天的数据(8941,12)
tmp_df['date'] = tmp_df['date'] - 2   # 将第一天改为第0天
tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True) #(201527,12)
tmp['date'] = tmp['date'].apply(get_refer_day)    # 29,2,3,

tmp.columns = ['ad_id','last2_num_click','date']   

all_data = all_data.merge(tmp, on=['ad_id','date'], how='left')

################################################################
def get_refer_day(d):
    return d + 3
    
tmp = all_data.copy()   #  (192586,12)
tmp = tmp[['ad_id','num_click','date']]

tmp_df = tmp[tmp.date<=2]  # 第一天的数据(8941,12)
tmp_df['date'] = tmp_df['date'] - 3   # 将第一天改为第0天
tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True) #(201527,12)
tmp['date'] = tmp['date'].apply(get_refer_day)    # 29,2,3,

tmp.columns = ['ad_id','last3_num_click','date']   

all_data = all_data.merge(tmp, on=['ad_id','date'], how='left')

all_data['last1_num_click'].fillna(0, inplace=True)
all_data['last2_num_click'].fillna(0, inplace=True)
all_data['last3_num_click'].fillna(0, inplace=True)

data = all_data[all_data['istest']==0]

test = all_data[all_data['istest']==1]

res = data[['ad_id','ad_bid','Delivery_time','num_click','Ad_account_id','date','week',
            'Commodity_id','Commodity_type','Ad_Industry_Id','Ad_material_size',
            'last1_num_click','last2_num_click','last3_num_click','istest']]

res_test = test[['ad_id','ad_bid','Delivery_time','num_click','Ad_account_id','date','week',
            'Commodity_id','Commodity_type','Ad_Industry_Id','Ad_material_size',
            'last1_num_click','last2_num_click','last3_num_click','istest']]

tmp = data.groupby(['Delivery_time'], as_index=False)['num_click'].agg({'Delivery_time_mean':'mean'})
res = res.merge(tmp, on=['Delivery_time'], how='left')
res_test = res_test.merge(tmp, on=['Delivery_time'], how='left')

tmp = data.groupby(['Ad_account_id'], as_index=False)['num_click'].agg({'Ad_account_id_mean':'mean'})
res = res.merge(tmp, on=['Ad_account_id'], how='left')
res_test = res_test.merge(tmp, on=['Ad_account_id'], how='left')

tmp = data.groupby(['Commodity_id'], as_index=False)['num_click'].agg({'Commodity_id_mean':'mean'})
res = res.merge(tmp, on=['Commodity_id'], how='left')
res_test = res_test.merge(tmp, on=['Commodity_id'], how='left')

tmp = data.groupby(['Commodity_type'], as_index=False)['num_click'].agg({'Commodity_type_mean':'mean'})
res = res.merge(tmp, on=['Commodity_type'], how='left')
res_test = res_test.merge(tmp, on=['Commodity_type'], how='left')

tmp = data.groupby(['Ad_Industry_Id'], as_index=False)['num_click'].agg({'Ad_Industry_Id_mean':'mean'})
res = res.merge(tmp, on=['Ad_Industry_Id'], how='left')
res_test = res_test.merge(tmp, on=['Ad_Industry_Id'], how='left')

tmp = data.groupby(['Ad_material_size'], as_index=False)['num_click'].agg({'Ad_material_size_mean':'mean'})
res = res.merge(tmp, on=['Ad_material_size'], how='left')
res_test = res_test.merge(tmp, on=['Ad_material_size'], how='left')

##########################################################################

tmp = data.groupby(['Ad_material_size','Ad_Industry_Id'], as_index=False)['num_click'].agg({'Ad_material_size&Ad_Industry_Id_mean':'mean'})
res['Ad_material_size&Ad_Industry_Id'] = res['Ad_material_size'].astype(str) + res['Ad_Industry_Id'].astype(str)
res_test['Ad_material_size&Ad_Industry_Id'] = res_test['Ad_material_size'].astype(str) + res_test['Ad_Industry_Id'].astype(str)
res = res.merge(tmp, on=['Ad_material_size','Ad_Industry_Id'], how='left')
res_test = res_test.merge(tmp, on=['Ad_material_size','Ad_Industry_Id'], how='left')

tmp = data.groupby(['Ad_material_size','Commodity_type'], as_index=False)['num_click'].agg({'material_size&Commodity_type_mean':'mean'})
res['material_size&Commodity_type'] = res['Ad_material_size'].astype(str) + res['Commodity_type'].astype(str)
res_test['material_size&Commodity_type'] = res_test['Ad_material_size'].astype(str) + res_test['Commodity_type'].astype(str)
res = res.merge(tmp, on=['Ad_material_size','Commodity_type'], how='left')
res_test = res_test.merge(tmp, on=['Ad_material_size','Commodity_type'], how='left')

tmp = data.groupby(['Ad_material_size','Commodity_id'], as_index=False)['num_click'].agg({'material_size&Commodity_id_mean':'mean'})
res['material_size&Commodity_id'] = res['Ad_material_size'].astype(str) + res['Commodity_id'].astype(str)
res_test['material_size&Commodity_id'] = res_test['Ad_material_size'].astype(str) + res_test['Commodity_id'].astype(str)
res = res.merge(tmp, on=['Ad_material_size','Commodity_id'], how='left')
res_test = res_test.merge(tmp, on=['Ad_material_size','Commodity_id'], how='left')

tmp = data.groupby(['Ad_material_size','Ad_account_id'], as_index=False)['num_click'].agg({'material_size&Ad_account_id_mean':'mean'})
res['material_size&Ad_account_id'] = res['Ad_material_size'].astype(str) + res['Ad_account_id'].astype(str)
res_test['material_size&Ad_account_id'] = res_test['Ad_material_size'].astype(str) + res_test['Ad_account_id'].astype(str)
res = res.merge(tmp, on=['Ad_material_size','Ad_account_id'], how='left')
res_test = res_test.merge(tmp, on=['Ad_material_size','Ad_account_id'], how='left')

tmp = data.groupby(['Ad_material_size','Delivery_time'], as_index=False)['num_click'].agg({'material_size&Delivery_time_mean':'mean'})
res['material_size&Delivery_time'] = res['Ad_material_size'].astype(str) + res['Delivery_time'].astype(str)
res_test['material_size&Delivery_time'] = res_test['Ad_material_size'].astype(str) + res_test['Delivery_time'].astype(str)
res = res.merge(tmp, on=['Ad_material_size','Delivery_time'], how='left')
res_test = res_test.merge(tmp, on=['Ad_material_size','Delivery_time'], how='left')

##################################################################################################
tmp = data.groupby(['Ad_Industry_Id','Commodity_type'], as_index=False)['num_click'].agg({'Ad_Industry_Id&Commodity_type_mean':'mean'})
res['Ad_Industry_Id&Commodity_type'] = res['Ad_Industry_Id'].astype(str) + res['Commodity_type'].astype(str)
res_test['Ad_Industry_Id&Commodity_type'] = res_test['Ad_Industry_Id'].astype(str) + res_test['Commodity_type'].astype(str)
res = res.merge(tmp, on=['Ad_Industry_Id','Commodity_type'], how='left')
res_test = res_test.merge(tmp, on=['Ad_Industry_Id','Commodity_type'], how='left')

tmp = data.groupby(['Ad_Industry_Id','Commodity_id'], as_index=False)['num_click'].agg({'Ad_Industry_Id&Commodity_id_mean':'mean'})
res['Ad_Industry_Id&Commodity_id'] = res['Ad_Industry_Id'].astype(str) + res['Commodity_id'].astype(str)
res_test['Ad_Industry_Id&Commodity_id'] = res_test['Ad_Industry_Id'].astype(str) + res_test['Commodity_id'].astype(str)
res = res.merge(tmp, on=['Ad_Industry_Id','Commodity_id'], how='left')
res_test = res_test.merge(tmp, on=['Ad_Industry_Id','Commodity_id'], how='left')

tmp = data.groupby(['Ad_Industry_Id','Ad_account_id'], as_index=False)['num_click'].agg({'Ad_Industry_Id&Ad_account_id_mean':'mean'})
res['Ad_Industry_Id&Ad_account_id'] = res['Ad_Industry_Id'].astype(str) + res['Ad_account_id'].astype(str)
res_test['Ad_Industry_Id&Ad_account_id'] = res_test['Ad_Industry_Id'].astype(str) + res_test['Ad_account_id'].astype(str)
res = res.merge(tmp, on=['Ad_Industry_Id','Ad_account_id'], how='left')
res_test = res_test.merge(tmp, on=['Ad_Industry_Id','Ad_account_id'], how='left')

tmp = data.groupby(['Ad_Industry_Id','Delivery_time'], as_index=False)['num_click'].agg({'Ad_Industry_Id&Delivery_time_mean':'mean'})
res['Ad_Industry_Id&Delivery_time'] = res['Ad_Industry_Id'].astype(str) + res['Delivery_time'].astype(str)
res_test['Ad_Industry_Id&Delivery_time'] = res_test['Ad_Industry_Id'].astype(str) + res_test['Delivery_time'].astype(str)
res = res.merge(tmp, on=['Ad_Industry_Id','Delivery_time'], how='left')
res_test = res_test.merge(tmp, on=['Ad_Industry_Id','Delivery_time'], how='left')

#############################################################################################
tmp = data.groupby(['Commodity_type','Ad_Industry_Id'], as_index=False)['num_click'].agg({'Commodity_type&industry_id_mean':'mean'})
res['Commodity_type&industry_id'] = res['Commodity_type'].astype(str) + res['Ad_Industry_Id'].astype(str)
res_test['Commodity_type&industry_id'] = res_test['Commodity_type'].astype(str) + res_test['Ad_Industry_Id'].astype(str)
res = res.merge(tmp, on=['Commodity_type','Ad_Industry_Id'], how='left')
res_test = res_test.merge(tmp, on=['Commodity_type','Ad_Industry_Id'], how='left')

tmp = data.groupby(['Commodity_type','Commodity_id'], as_index=False)['num_click'].agg({'Commodity_type&Commodity_id_mean':'mean'})
res['Commodity_type&Commodity_id'] = res['Commodity_type'].astype(str) + res['Commodity_id'].astype(str)
res_test['Commodity_type&Commodity_id'] = res_test['Commodity_type'].astype(str) + res_test['Commodity_id'].astype(str)
res = res.merge(tmp, on=['Commodity_type','Commodity_id'], how='left')
res_test = res_test.merge(tmp, on=['Commodity_type','Commodity_id'], how='left')

tmp = data.groupby(['Commodity_type','Ad_account_id'], as_index=False)['num_click'].agg({'Commodity_type&Ad_account_id_mean':'mean'})
res['Commodity_type&Ad_account_id'] = res['Commodity_type'].astype(str) + res['Ad_account_id'].astype(str)
res_test['Commodity_type&Ad_account_id'] = res_test['Commodity_type'].astype(str) + res_test['Ad_account_id'].astype(str)
res = res.merge(tmp, on=['Commodity_type','Ad_account_id'], how='left')
res_test = res_test.merge(tmp, on=['Commodity_type','Ad_account_id'], how='left')

tmp = data.groupby(['Commodity_type','Delivery_time'], as_index=False)['num_click'].agg({'Commodity_type&Delivery_time_mean':'mean'})
res['Commodity_type&Delivery_time'] = res['Commodity_type'].astype(str) + res['Delivery_time'].astype(str)
res_test['Commodity_type&Delivery_time'] = res_test['Commodity_type'].astype(str) + res_test['Delivery_time'].astype(str)
res = res.merge(tmp, on=['Commodity_type','Delivery_time'], how='left')
res_test = res_test.merge(tmp, on=['Commodity_type','Delivery_time'], how='left')

######################################################################################################
tmp = data.groupby(['Commodity_id','Ad_account_id'], as_index=False)['num_click'].agg({'Commodity_id&Ad_account_id_mean':'mean'})
res['Commodity_id&Ad_account_id'] = res['Commodity_id'].astype(str) + res['Ad_account_id'].astype(str)
res_test['Commodity_id&Ad_account_id'] = res_test['Commodity_id'].astype(str) + res_test['Ad_account_id'].astype(str)
res = res.merge(tmp, on=['Commodity_id','Ad_account_id'], how='left')
res_test = res_test.merge(tmp, on=['Commodity_id','Ad_account_id'], how='left')

tmp = data.groupby(['Commodity_id','Delivery_time'], as_index=False)['num_click'].agg({'Commodity_id&Delivery_time_mean':'mean'})
res['Commodity_id&Delivery_time'] = res['Commodity_id'].astype(str) + res['Delivery_time'].astype(str)
res_test['Commodity_id&Delivery_time'] = res_test['Commodity_id'].astype(str) + res_test['Delivery_time'].astype(str)
res = res.merge(tmp, on=['Commodity_id','Delivery_time'], how='left')
res_test = res_test.merge(tmp, on=['Commodity_id','Delivery_time'], how='left')

#################################################################################
tmp = data.groupby(['Ad_account_id','Delivery_time'], as_index=False)['num_click'].agg({'Ad_account_id&Delivery_time_mean':'mean'})
res['Ad_account_id&Delivery_time'] = res['Ad_account_id'].astype(str) + res['Delivery_time'].astype(str)
res_test['Ad_account_id&Delivery_time'] = res_test['Ad_account_id'].astype(str) + res_test['Delivery_time'].astype(str)
res = res.merge(tmp, on=['Ad_account_id','Delivery_time'], how='left')
res_test = res_test.merge(tmp, on=['Ad_account_id','Delivery_time'], how='left')



res_test = res_test.fillna(res_test.mean())

all_res = pd.concat( [res, res_test], axis=0)

from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

['ad_id','ad_bid','Delivery_time','num_click','Ad_account_id', 
            'Commodity_id','Commodity_type','Ad_Industry_Id','Ad_material_size',
            'last1_num_click','last2_num_click','last3_num_click','istest']

all_res['Delivery_time'] = lbl.fit_transform(all_res['Delivery_time'].astype(str))
all_res['Ad_account_id'] = lbl.fit_transform(all_res['Ad_account_id'].astype(str))
all_res['Commodity_id'] = lbl.fit_transform(all_res['Commodity_id'].astype(str))
all_res['Commodity_type'] = lbl.fit_transform(all_res['Commodity_type'].astype(str))
all_res['Ad_Industry_Id'] = lbl.fit_transform(all_res['Ad_Industry_Id'].astype(str))
all_res['Ad_material_size'] = lbl.fit_transform(all_res['Ad_material_size'].astype(str))

all_res['Ad_material_size&Ad_Industry_Id'] = lbl.fit_transform(all_res['Ad_material_size&Ad_Industry_Id'].astype(str))
all_res['material_size&Commodity_type'] = lbl.fit_transform(all_res['material_size&Commodity_type'].astype(str))
all_res['material_size&Commodity_id'] = lbl.fit_transform(all_res['material_size&Commodity_id'].astype(str))
all_res['material_size&Ad_account_id'] = lbl.fit_transform(all_res['material_size&Ad_account_id'].astype(str))
all_res['material_size&Delivery_time'] = lbl.fit_transform(all_res['material_size&Delivery_time'].astype(str))
all_res['Ad_Industry_Id&Commodity_type'] = lbl.fit_transform(all_res['Ad_Industry_Id&Commodity_type'].astype(str))
all_res['Ad_Industry_Id&Commodity_id'] = lbl.fit_transform(all_res['Ad_Industry_Id&Commodity_id'].astype(str))
all_res['Ad_Industry_Id&Ad_account_id'] = lbl.fit_transform(all_res['Ad_Industry_Id&Ad_account_id'].astype(str))
all_res['Ad_Industry_Id&Delivery_time'] = lbl.fit_transform(all_res['Ad_Industry_Id&Delivery_time'].astype(str))
all_res['Commodity_type&industry_id'] = lbl.fit_transform(all_res['Commodity_type&industry_id'].astype(str))
all_res['Commodity_type&Commodity_id'] = lbl.fit_transform(all_res['Commodity_type&Commodity_id'].astype(str))
all_res['Commodity_type&Ad_account_id'] = lbl.fit_transform(all_res['Commodity_type&Ad_account_id'].astype(str))
all_res['Commodity_type&Delivery_time'] = lbl.fit_transform(all_res['Commodity_type&Delivery_time'].astype(str))
all_res['Commodity_id&Ad_account_id'] = lbl.fit_transform(all_res['Commodity_id&Ad_account_id'].astype(str))
all_res['Commodity_id&Delivery_time'] = lbl.fit_transform(all_res['Commodity_id&Delivery_time'].astype(str))
all_res['Ad_account_id&Delivery_time'] = lbl.fit_transform(all_res['Ad_account_id&Delivery_time'].astype(str))

train_res = all_res[all_res['istest']==0]
train_res.to_csv('./train/train_hebing.csv',index=None)

test_res = all_res[all_res['istest']==1]
test_res.to_csv('./train/test_hebing.csv',index=None)
#res.to_csv('./train/train_last3.csv',index=None)