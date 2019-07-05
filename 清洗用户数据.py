# -*- coding: utf-8 -*-
import pandas as pd
userFeature_data = []
 
user_Feature_columns = ['user_id', 'Age', 'Gender', 'Area', 'Marriage_Status', 'Education', 
                        'Consuption_Ability', 'Device','Work_Status', 'Connection_Type', 'Behavior']

userFeature_data.append(user_Feature_columns)

with open('/cos_public/cephfs/tesla_common/deeplearning/dataset/AI_Race/user_data/user_data.out', 'r') as f:
    for i, line in enumerate(f):
        line = line.strip().split('\t')
        userFeature_data.append(line)

user_feature = pd.DataFrame(userFeature_data)
user_feature.to_csv('/cos_person/tencent/train/userFeature.csv', index=False, header=False)
 
 