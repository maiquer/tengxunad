# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:58:29 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt

rootPath = r'F:\腾讯广告算法大赛\复赛\Gr'
# 0： 广告ID：共有259228条广告id，25.9w   清洗后剩余254772
# 1： 创建时间：删除掉创建时间为-1和0的广告           ##############需清洗
# 2:  广告账户id ：共12285，1w个无重复的
# 3： 商品id：共7672，0.7w个无重复
# 4： 商品类型   共7种1,2,3,4,5,6,7
# 5： 广告行业id: 198种：-1~197
# 6： 素材尺寸  有出现-1,没有出现同一个广告有多个不同尺寸的素材的情况   ###########需清洗
ad_static =[]
#re = []
with open(rootPath + r'\map_ad_static.out', 'r') as f:
    for i, line in enumerate(f):  
        line = line.strip().split('\t')
        ad_static.append(line)
        #re.append(int(line[0]))

        
## 商品类型统计部分
#c = Counter(re)
#cc =c.most_common()
#ccc = np.array(cc)
#plt.bar(ccc[:,0], ccc[:,1], label='graph 1')  
        

 
ad_static_cloumns = ['ad_id','create_time','account_id',' commodity_id',
                     'commodity_type','industry_id','Material_size']   

ad_static_df = pd.DataFrame(ad_static,columns = ad_static_cloumns)    


# 清洗掉广告创建时间为0和1的广告 -0.4w
illegality_create = ['0','-1']
ad_static_df = ad_static_df[~ad_static_df['create_time'].isin(illegality_create)]
 
# 清洗掉广告素材尺寸为-1的广告
ad_static_df = ad_static_df[ad_static_df['Material_size']!='-1']
def time_change(x):
    x = time.localtime(int(x))
    x = time.strftime("%Y-%m-%d %H:%M:%S", x)
    return x


ad_static_df['create_time'] = ad_static_df['create_time'].apply(time_change)

ad_static_df['ad_id'] = ad_static_df['ad_id'].astype(int)
ad_static_df['account_id'] = ad_static_df['account_id'].astype(int)
ad_static_df['commodity_id'] = ad_static_df[' commodity_id'].astype(int)
ad_static_df['commodity_type'] = ad_static_df['commodity_type'].astype(int)
ad_static_df['industry_id'] = ad_static_df['industry_id'].astype(int)
ad_static_df['Material_size'] = ad_static_df['Material_size'].astype(int)
ad_static_df.to_csv('ad_static.csv',index = None)

