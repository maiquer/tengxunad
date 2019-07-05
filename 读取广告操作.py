# -*- coding: utf-8 -*-
import os
import sys
import pip
# 设置日志实时刷新
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
os.system('free -h') # 查看内存分配情况
pip.main(['install', 'lightgbm']) # 安装包

# 主代码
import pandas as pd
import lightgbm as lgb
path = '${ai_dataset_lib}/AI_Race/'
ad = pd.read_csv(path + 'ads_data/map_ad_static.out') # 读取数据
print('ad:')
print(ad.head()) 
