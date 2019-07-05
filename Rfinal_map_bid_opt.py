# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:12:38 2019

@author: maiquer
"""

import pandas as pd
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt

rootPath = r'F:\腾讯广告算法大赛\复赛\Gr'
ad_static = pd.read_csv('ad_static.csv')
static_adid = list(ad_static['ad_id'].values)
opt_columns = ['ad_id','modify_time','opt_type','target_changeType','charge_type','new_bid']
# 0： 广告ID：共有321753, 32w条广告id,去除重复之后有221951,22w   ###########可以通过static清洗部分
# 1： 创建/修改时间： 2019/04/10/00/00/00 - 2019/04/22/23/59/53           
# 2:  操作类型两种：1代表修改；2代表新建
# 3： 新的目标转化类型：0为传统竞价广告，211 -1~13 大于0代表ocpa广告的不同转化目标  ############其中有部分数据为-1，需清洗
# 4： 操作后新的计费类型  0,1,2 0有215个  1按照展现计费，2按照点击计费   ############清洗掉计费类型为0的数据
# 5： 操作后新的出价值: 10~20000,10580种

ad_opt =[]
re = []
with open(rootPath + r'\final_map_bid_opt.out', 'r') as f:
    for i, line in enumerate(f):  
        line = line.strip().split('\t')
        ad_opt.append(line)
        re.append(int(line[0]))

  
     
xx = list(set(re))
xxx = np.array(xx)


c = Counter(re)
cc =c.most_common(100)
ccc = np.array(cc)
#plt.bar(ccc[:,0], ccc[:,1], label='graph 1')  
ad_opt_df = pd.DataFrame(ad_opt,columns = opt_columns)

def change_time(x):
    return x[:4]+'-'+ x[4:6] + '-'+ x[6:8]+'-'+x[8:10]+':'+x[10:12]+':'+x[12:14]

test = ad_opt_df['modify_time'].apply(change_time)
### 通过广告id清洗
ad_opt_df['ad_id'] = ad_opt_df['ad_id'].astype(int)
# 321753-> 319792
ad_opt_df = ad_opt_df[ad_opt_df['ad_id'].isin(static_adid)]
#new1 = ad_opt_df[~ad_opt_df['ad_id'].isin(static_adid)]


##### 通过目标转化类型清洗数据,其实不符合条件的ad已经通过静态广告过滤掉了
#ad_opt_df['target_changeType'] = ad_opt_df['target_changeType'].astype(int)
ad_opt_df = ad_opt_df[ad_opt_df['target_changeType']!= '-1']


#####通过计费类型清洗数据
ad_opt_df = ad_opt_df[ad_opt_df['charge_type']!='0']

ad_opt_df.to_csv('ad_opt_df.csv',index = None)

# eg.同一个广告在不同的时间会修改不同的报价，但是目标转化类型和计费类型均不变
ad201649 = ad_opt_df[ad_opt_df['ad_id']==201649]
ad201649 = ad201649.sort_values(by = 'modify_time')
ad201649['modify_time'] = ad201649['modify_time'].apply(change_time)


ad137478 = ad_opt_df[ad_opt_df['ad_id']== 137478]
ad137478 = ad137478.sort_values(by = 'modify_time')
ad137478['modify_time'] = ad137478['modify_time'].apply(change_time)