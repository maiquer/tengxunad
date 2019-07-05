# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:05:45 2019

@author: Mr
"""

import pandas as pd

Ad_Static_Feature_Data = pd.read_csv('./train/Ad_Static_Feature_Data.csv', header=0)
res = Ad_Static_Feature_Data.loc[(Ad_Static_Feature_Data['ad_id'] == 198587)]