#!/usr/bin/env python
# coding: utf-8

# # 제품 이상여부 판별 프로젝트 (본선)
# 

# ## 1. 데이터 불러오기
# 

# ### 필수 라이브러리
# 

# In[1]:


get_ipython().system('pip install hyperopt lightgbm')


# In[2]:


import os
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# 모든 열 출력
pd.set_option('display.max_columns', None)


# ### 데이터 읽어오기
# 

# In[4]:


ROOT_DIR = "data"
RANDOM_STATE = 110

# Load data
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
train_data


# In[5]:


test = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
test


# ## 2. 수기 데이터 읽어오기

# 수기 데이터는 제조 환경과 관련된 정보들이 담겨있는 데이터입니다.

# In[6]:


df_hand = pd.read_excel(os.path.join(ROOT_DIR, "hand_data.xlsx"))
df_hand


# In[7]:


drop_col=[]
for col in train_data.columns:
  nullcount = train_data[col].isnull().sum()
  nunique = train_data[col].nunique()
  if nullcount == len(train_data):
    drop_col.append(col)
  if (nunique==1) & (nullcount == 0):
    drop_col.append(col)
    
train_data = train_data.drop(drop_col, axis=1)
test = test.drop(drop_col, axis=1)


train_data


# In[8]:


cols=["Model.Suffix_Dam","Model.Suffix_AutoClave","Model.Suffix_Fill1","Model.Suffix_Fill2"]

train_data["Model.Suffix"]= train_data["Model.Suffix_Dam"]
train_data = train_data.drop(cols, axis=1)

cols=["Model.Suffix_Dam","Model.Suffix_AutoClave","Model.Suffix_Fill1","Model.Suffix_Fill2"]

test["Model.Suffix"]= test["Model.Suffix_Dam"]
test = test.drop(cols, axis=1)


cols=["Workorder_Dam","Workorder_AutoClave","Workorder_Fill1","Workorder_Fill2"]

train_data["Workorder"]= train_data["Workorder_Dam"]
train_data = train_data.drop(cols, axis=1)

cols=["Workorder_Dam","Workorder_AutoClave","Workorder_Fill1","Workorder_Fill2"]

test["Workorder"]= test["Workorder_Dam"]
test = test.drop(cols, axis=1)


# In[9]:


train_data['running_time'] = pd.to_datetime(train_data['Collect Date_AutoClave']) - pd.to_datetime(train_data['Collect Date_Dam'])
train_data['running_time_m'] = train_data['running_time'].dt.total_seconds() / 60


# In[10]:


train_data['Collect Date_Dam'] = pd.to_datetime(train_data['Collect Date_Dam'])
train_data['month'] = train_data['Collect Date_Dam'].dt.month
train_data['day'] = train_data['Collect Date_Dam'].dt.day
train_data['hour'] = train_data['Collect Date_Dam'].dt.hour


# In[11]:


train_data['out_of_hour'] = 0  # 기본값으로 0 설정
train_data.loc[(train_data['hour'] < 8) | (train_data['hour'] > 20), 'out_of_hour'] = 1  # 조건을 만족하는 경우 a열에 1을 할당


# In[12]:


train_data


# In[13]:


train_data = train_data.drop(columns='running_time', axis=1)


# In[14]:


test['running_time'] = pd.to_datetime(test['Collect Date_AutoClave']) - pd.to_datetime(test['Collect Date_Dam'])
test['running_time_m'] = test['running_time'].dt.total_seconds() / 60


# In[15]:


test['Collect Date_Dam'] = pd.to_datetime(test['Collect Date_Dam'])
test['month'] = test['Collect Date_Dam'].dt.month
test['day'] = test['Collect Date_Dam'].dt.day
test['hour'] = test['Collect Date_Dam'].dt.hour


# In[16]:


test['out_of_hour'] = 0  # 기본값으로 0 설정
test.loc[(test['hour'] < 8) | (test['hour'] > 20), 'out_of_hour'] = 1  # 조건을 만족하는 경우 a열에 1을 할당


# In[17]:


test = test.drop(columns='running_time', axis=1)


# In[18]:


train_data


# In[19]:


train_head=[]
# 그룹핑
for column in train_data.columns:
  if 'head' in column.lower():
    train_head.append(column)
train_head_data=train_data[train_head]
    


# In[20]:


train_head_data


# In[21]:


nan_data=['HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam',
'GMES_ORIGIN_INSP_JUDGE_CODE Collect Result_AutoClave',
'GMES_ORIGIN_INSP_JUDGE_CODE Judge Value_AutoClave',
'HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Fill1',
'HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Fill2',]
nanana=train_data[nan_data]


# In[22]:


# 첫 번째 컬럼을 기준으로 나머지 컬럼들과 비교
first_col = nanana.iloc[:, 0]
comparison_results = {}

# 각 컬럼에 대해 비교 결과 저장
for col in nanana.columns[1:]:
    comparison_results[col] = first_col.equals(nanana[col])

# 결과 출력
for col, is_equal in comparison_results.items():
    if is_equal:
        train_data=train_data.drop(col, axis=1)
        test=test.drop(col,axis=1)
        print(f"컬럼 '{nan_data[0]}'와 컬럼 '{col}'는 동일합니다.")
    else:
        print(f"컬럼 '{nan_data[0]}'와 컬럼 '{col}'는 다릅니다.")


# In[23]:


train_data


# In[24]:


train_data['HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam'] = train_data['HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam'].combine_first(
    train_data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'].apply(lambda x: 'nan' if pd.isna(x) else np.nan)
)
# 'Collect Result_Dam' 값이 숫자인 경우 'Judge Value_Dam'를 'NG'로 채우기
train_data.loc[
    (pd.to_numeric(train_data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'], errors='coerce').notna()) &
    (~train_data['HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam'].isin(['OK', 'NaN'])),
    'HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam'
] = 'NG'


test['HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam'] = test['HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam'].combine_first(
    test['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'].apply(lambda x: 'nan' if pd.isna(x) else np.nan)
)
# 'Collect Result_Dam' 값이 숫자인 경우 'Judge Value_Dam'를 'NG'로 채우기
test.loc[
    (pd.to_numeric(test['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'], errors='coerce').notna()) &
    (~test['HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam'].isin(['OK', 'NaN'])),
    'HEAD NORMAL COORDINATE X AXIS(Stage1) Judge Value_Dam'
] = 'NG'


# In[25]:


train_data


# In[26]:


train_data.info(200)


# In[27]:


# 'OK' 값을 NaN으로 변환하고 결측치 처리할 열 리스트
columns_to_replace = [
    'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam',
    'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1',
    'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2'
]
# 결측치를 처리할 열 리스트
columns_to_fill = [
    'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam',
    'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Dam',
    'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Dam',
    'HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Dam',
    'HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Dam',
    'HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Dam',
    'HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Dam',
    'HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Dam',
    'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1',
    'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2'
]


# In[28]:


def preprocess_coordinates(df, columns_to_replace, columns_to_fill):
    def replace_value(row, column, possible_values, mapping, stage2_col, stage3_col):
        # Stage2와 Stage3의 값 처리
        stage2_value = row[stage2_col]
        stage2_value = int(str(round(stage2_value,1))[:1])
        stage3_value = round(row[stage3_col])
        stage3_value = int(str(stage3_value)[:1])

        # 선택 가능한 값
        remaining_value = [val for val in possible_values if val not in [stage2_value, stage3_value]]
        if len(remaining_value) == 1:
            replacement_value = mapping[remaining_value[0]]
            return replacement_value
        else:
            return row[column]
    def get_mapping(val):
        val = int(val) // 100
        if val in [3, 5, 6]:  # 논리 연산자 사용
            return  [3, 5, 6], {3: 305.0, 5: 499.8, 6: 694.0}
        else:
            return [1, 4, 8],{1: 156.0, 4: 458.0, 8: 835.8}

    # 1. 특정 값('OK')을 NaN으로 변환
    for column in columns_to_replace:
        df[column].replace('OK', np.nan, inplace=True)
    # 2. 숫자로 변환할 수 없는 데이터를 NaN으로 변환
    for column in columns_to_replace:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # 3. 결측치를 처리하기 위한 최빈값 계산 및 채우기
    for column in columns_to_fill:
        if 'Dam' in column:
            possible_values = [1, 4, 5]
            stage2_col = 'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Dam'
            stage3_col = 'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Dam'
            mapping = {1: 161.0, 4: 464.0, 5: 550.4}
            mask = df[column].isna()
            df.loc[mask, column] = df[mask].apply(
                lambda row: replace_value(row, column, possible_values, mapping, stage2_col, stage3_col),
                axis=1
            )
        if 'Fill1' in column:
            possible_values = [1, 4, 8]
            stage2_col = 'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1'
            stage3_col = 'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill1'
            mapping = {1: 157.0, 4: 458.5, 8: 838.4}
            mask = df[column].isna()
            df.loc[mask, column] = df[mask].apply(
                lambda row: replace_value(row, column, possible_values, mapping, stage2_col, stage3_col),
                axis=1
            )

        if 'Fill2' in column:
            stage2_col = 'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill2'
            stage3_col = 'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill2'

            mask = df[column].isna()
            df.loc[mask, column] = df[mask].apply(lambda row: replace_value(row, column, *get_mapping(round(row[stage2_col])), stage2_col, stage3_col),axis=1)

    return df


# In[29]:


# 결측치 함수
train_data = preprocess_coordinates(train_data, columns_to_replace, columns_to_fill)
test=preprocess_coordinates(test, columns_to_replace, columns_to_fill)


# In[30]:


train_data.loc[train_data["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1"] == 681.2, 'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1'] = 838.0
train_data.loc[train_data["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1"] == 681.2, 'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill1'] = 157.0
train_data.loc[train_data["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1"] == 681.2, 'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1'] = 458.5

test.loc[test["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1"] == 681.2, 'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1'] = 838.0
test.loc[test["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1"] == 681.2, 'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill1'] = 157.0
test.loc[test["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1"] == 681.2, 'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1'] = 458.5


# In[31]:


x = train_data.drop('target', axis=1)
y = train_data['target']


# In[32]:


train_head=[]
# 그룹핑
for column in train_data.columns:
  if 'head' in column.lower():
    train_head.append(column)
train_head_data=train_data[train_head]


# In[33]:


train_head_data


# In[34]:


pos_col = train_head_data.columns.copy()

grouped_columns = {}
grouped_columns['Dam'] = {}
grouped_columns['Fill1'] = {}
grouped_columns['Fill2'] = {}

# 그룹핑
for column in pos_col:
  if 'judge' not in column.lower():
    parts = column.split('_')
    if len(parts) == 2:
        coord_info, process_type = parts
        axis = 'X' if ' X ' in coord_info else 'Y' if ' Y ' in coord_info else 'Z' if ' Z ' in coord_info else 'Θ' if ' Θ ' in coord_info else 'Unknown'
        # axis가 존재하지 않으면 새로 생성
        if axis not in grouped_columns[process_type]:
            grouped_columns[process_type][axis] = []
        grouped_columns[process_type][axis].append(column)

# 딕셔너리의 각 프로세스와 축에 대해 새로운 컬럼 생성
for process_type, axes in grouped_columns.items():
    for axis, columns in axes.items():
        # 각 축에 대한 컬럼을 tuple로 묶어 새로운 컬럼을 생성
        new_column_name = f'{process_type} {axis}'  # ex: 'Dam X', 'Fill1 X'
        train_data[new_column_name] = list(zip(*[train_data[col] for col in columns]))
        train_data[new_column_name] = train_data[new_column_name].astype(str)
        test[new_column_name] = list(zip(*[test[col] for col in columns]))
        test[new_column_name] = test[new_column_name].astype(str)


# In[35]:


train_data


# In[36]:


cleanpurge_col_set = [['Head Clean Position X Collect Result_Fill1',
'Head Purge Position X Collect Result_Fill1'],
                 ['Head Clean Position Y Collect Result_Fill1',
'Head Purge Position Y Collect Result_Fill1'],
                 ['Head Clean Position Z Collect Result_Fill1',
'Head Purge Position Z Collect Result_Fill1'],
                 
                 ['Head Clean Position X Collect Result_Dam',
'Head Purge Position X Collect Result_Dam'],
                 ['Head Clean Position Y Collect Result_Dam',
'Head Purge Position Y Collect Result_Dam'],
                 ['Head Clean Position Z Collect Result_Dam',
'Head Purge Position Z Collect Result_Dam']]

new_col = []
for col_set in cleanpurge_col_set:
  colname = col_set[0].replace('Clean', 'Clean&Purge')
  new_col.append(colname)
  train_data[colname] = list(train_data[col_set].itertuples(index=False, name=None))
  train_data[colname] = train_data[colname].astype(str)
  test[colname] = list(test[col_set].itertuples(index=False, name=None))
  test[colname] = test[colname].astype(str)


# In[37]:


train_data


# In[38]:


cure_col_set = [ ['CURE START POSITION X Collect Result_Dam',
                  'CURE END POSITION X Collect Result_Dam'],
                 ['CURE START POSITION Θ Collect Result_Dam',
                  'CURE END POSITION Θ Collect Result_Dam'],
                 [ 'CURE START POSITION X Collect Result_Fill2',
                   'CURE END POSITION X Collect Result_Fill2'],
                 [ 'CURE START POSITION Z Collect Result_Fill2',
                   'CURE END POSITION Z Collect Result_Fill2']]



new_col = []
for col_set in cure_col_set:
  colname = col_set[0].replace('START', '')
  new_col.append(colname)
  train_data[colname] = list(train_data[col_set].itertuples(index=False, name=None))
  train_data[colname] = train_data[colname].astype(str)
  test[colname] = list(test[col_set].itertuples(index=False, name=None))
  test[colname] = test[colname].astype(str)


# In[39]:


stage_col_set = [['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam',
                  'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Dam',
                  'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Dam'],
                 ['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Dam',
                  'HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Dam',
                  'HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Dam'],
                 ['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Dam',
                  'HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Dam',
                  'HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Dam'],
                 ['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1',
                  'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1',
                  'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill1'],
                 ['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill1',
                  'HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill1',
                  'HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill1'],
                 ['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill1',
                  'HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill1',
                  'HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill1'],
                 ['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2',
                  'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill2',
                  'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill2'],
                 ['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill2',
                  'HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill2',
                  'HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill2'],
                 ['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill2',
                  'HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill2',
                  'HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill2']]
new_col = []
for col_set in stage_col_set:
  colname = col_set[0].replace('1)', ')')
  new_col.append(colname)
  train_data[colname] = list(train_data[col_set].itertuples(index=False, name=None))
  train_data[colname] = train_data[colname].astype(str)
  test[colname] = list(test[col_set].itertuples(index=False, name=None))
  test[colname] = test[colname].astype(str)


# In[40]:


import re
coordinate_groups = {}

head_dam_XYZ_col = []
head_fill1_XYZ_col = []
head_fill2_XYZ_col = []

cure_XYZ_col = []

# filtered_columns = [col for col in column_names if 'CURE' not in col]




for col in pos_col:
    # 정규 표현식을 사용하여 X, Y, Z 부분을 찾고, 공통 이름을 추출
    match = re.match(r'(.*?)(\bX\b|\bY\b|\bZ\b)(.*)', col)
    if match:
        base_name = match.group(1).strip() + ' ' + match.group(3).strip()
        if base_name not in coordinate_groups:
            coordinate_groups[base_name] = {}
        coordinate_groups[base_name][match.group(2)] = col

# 각 그룹에 대해 가능한 X, Y, Z 좌표를 묶어서 새로운 컬럼 추가
for base_name, coords in coordinate_groups.items():
    # 기존 컬럼 중 존재하는 X, Y, Z 값을 가져와서 묶기
    grouped_columns_train = []
    grouped_columns_test = []
    for coord in ['X', 'Y', 'Z']:
        if coord in coords:
            grouped_columns_train.append(train_data[coords[coord]])
            grouped_columns_test.append(test[coords[coord]])
        else:
            grouped_columns_train.append([None] * len(train_data))  # 해당 컬럼이 없으면 None으로 채움
            grouped_columns_test.append([None] * len(test))
    # 새로운 (X, Y, Z) 튜플 컬럼 생성
    train_data[f'{base_name} XYZ'] = [str(tup) for tup in list(zip(*grouped_columns_train))] 
    test[f'{base_name} XYZ'] = [str(tup) for tup in list(zip(*grouped_columns_test))]
    if 'head' in base_name.lower():
        if 'dam' in base_name.lower():
            head_dam_XYZ_col.append(f'{base_name} XYZ')
        if 'fill1' in base_name.lower():
            head_fill1_XYZ_col.append(f'{base_name} XYZ')
        if 'fill2' in base_name.lower():
            head_fill2_XYZ_col.append(f'{base_name} XYZ')
    
            
    # else:
    #     cure_XYZ_col.append(f'{base_name} XYZ')
    
    # 기존의 X, Y, Z 컬럼 제거
    train_data.drop(columns=list(coords.values()), inplace=True)
    test.drop(columns=list(coords.values()), inplace=True)


# In[41]:


train_data


# In[42]:


head_dam_XYZ_col


# In[43]:


del head_dam_XYZ_col[1]  #Judge 제외
del head_dam_XYZ_col[-1] # Zero 제외
 

head_XYZ_col = head_dam_XYZ_col + head_fill1_XYZ_col + head_fill2_XYZ_col 

head_dam_XYZ_col = [head_dam_XYZ_col[3]] + head_dam_XYZ_col[:3] + head_dam_XYZ_col[4:]
head_fill1_XYZ_col = [head_fill1_XYZ_col[3]] + head_fill1_XYZ_col[:3] + head_fill1_XYZ_col[4:]
head_fill2_XYZ_col = [head_fill2_XYZ_col[3]] + head_fill2_XYZ_col[:3] + head_fill2_XYZ_col[4:]


# In[44]:


head_dam_XYZ_col


# In[ ]:





# In[45]:


from sklearn.preprocessing import LabelEncoder

# 1. 라벨 인코딩 적용
label_encoders = {}
for col in head_dam_XYZ_col:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    for label in list(test[col].unique()): #train에 없는 test값 처리
        if label not in list(le.classes_):
            le.classes_ = np.append(le.classes_,label)
    test[col] = le.transform(test[col])
    label_encoders[col] = le
    train_data[col] = train_data[col].astype(str) # 범주 변수 처리
    test[col] = test[col].astype(str) # 범주 변수 처리
    
# 2. 전체 상호작용항 생성 및 라벨 인코딩
interaction_col_name = 'head_interactionDam'  # 향후에 head_interaction dam/fill1/fill2로 분리해 볼 필요 있음

train_data[interaction_col_name] = train_data[head_dam_XYZ_col].astype(str).agg('_'.join, axis=1)
test[interaction_col_name] = test[head_dam_XYZ_col].astype(str).agg('_'.join, axis=1)


# In[46]:


from sklearn.preprocessing import LabelEncoder

# 1. 라벨 인코딩 적용
label_encoders = {}
for col in head_fill1_XYZ_col:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    for label in list(test[col].unique()): #train에 없는 test값 처리
        if label not in list(le.classes_):
            le.classes_ = np.append(le.classes_,label)
    test[col] = le.transform(test[col])
    label_encoders[col] = le
    train_data[col] = train_data[col].astype(str) # 범주 변수 처리
    test[col] = test[col].astype(str) # 범주 변수 처리
    
# 2. 전체 상호작용항 생성 및 라벨 인코딩
interaction_col_name = 'head_interactionFill1'  # 향후에 head_interaction dam/fill1/fill2로 분리해 볼 필요 있음

train_data[interaction_col_name] = train_data[head_fill1_XYZ_col].astype(str).agg('_'.join, axis=1)
test[interaction_col_name] = test[head_fill1_XYZ_col].astype(str).agg('_'.join, axis=1)


# In[47]:


from sklearn.preprocessing import LabelEncoder

# 1. 라벨 인코딩 적용
label_encoders = {}
for col in head_fill2_XYZ_col:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    for label in list(test[col].unique()): #train에 없는 test값 처리
        if label not in list(le.classes_):
            le.classes_ = np.append(le.classes_,label)
    test[col] = le.transform(test[col])
    label_encoders[col] = le
    train_data[col] = train_data[col].astype(str) # 범주 변수 처리
    test[col] = test[col].astype(str) # 범주 변수 처리
    
# 2. 전체 상호작용항 생성 및 라벨 인코딩
interaction_col_name = 'head_interactionFill2'  # 향후에 head_interaction dam/fill1/fill2로 분리해 볼 필요 있음

train_data[interaction_col_name] = train_data[head_fill2_XYZ_col].astype(str).agg('_'.join, axis=1)
test[interaction_col_name] = test[head_fill2_XYZ_col].astype(str).agg('_'.join, axis=1)


# In[ ]:





# In[48]:


from sklearn.preprocessing import LabelEncoder

# # 1. 라벨 인코딩 적용
# label_encoders = {}
# for col in head_XYZ_col:
#     le = LabelEncoder()
#     train_data[col] = le.fit_transform(train_data[col])
#     for label in list(test[col].unique()): #train에 없는 test값 처리
#         if label not in list(le.classes_):
#             le.classes_ = np.append(le.classes_,label)
#     test[col] = le.transform(test[col])
#     label_encoders[col] = le
#     train_data[col] = train_data[col].astype(str) # 범주 변수 처리
#     test[col] = test[col].astype(str) # 범주 변수 처리
    
# # 2. 전체 상호작용항 생성 및 라벨 인코딩
# interaction_col_name = 'head_interaction'  # 향후에 head_interaction dam/fill1/fill2로 분리해 볼 필요 있음

# train_data[interaction_col_name] = train_data[head_XYZ_col].astype(str).agg('_'.join, axis=1)
# test[interaction_col_name] = test[head_XYZ_col].astype(str).agg('_'.join, axis=1)

# 2. 순차적 상호작용

train_data['stage123_dam'] = train_data[head_dam_XYZ_col].iloc[:,1] + '_' + train_data[head_dam_XYZ_col].iloc[:,2] + '_' + train_data[head_dam_XYZ_col].iloc[:,3]
train_data['clean_purge_dam'] = train_data[head_dam_XYZ_col].iloc[:,4] + '_' + train_data[head_dam_XYZ_col].iloc[:,4] 

train_data['stage123_fill1'] = train_data[head_fill1_XYZ_col].iloc[:,1] + '_' + train_data[head_fill1_XYZ_col].iloc[:,2] + '_' + train_data[head_fill1_XYZ_col].iloc[:,3]
train_data['clean_purge_fill1'] = train_data[head_fill1_XYZ_col].iloc[:,4] + '_' + train_data[head_fill1_XYZ_col].iloc[:,5] 

train_data['stage123_fill2'] = train_data[head_fill2_XYZ_col].iloc[:,1] + '_' + train_data[head_fill2_XYZ_col].iloc[:,2] + '_' + train_data[head_fill2_XYZ_col].iloc[:,3]
train_data['clean_purge_fill2'] = train_data[head_fill2_XYZ_col].iloc[:,4] + '_' + train_data[head_fill2_XYZ_col].iloc[:,5] 

####################################################################################################################3

test['stage123_dam'] = test[head_dam_XYZ_col].iloc[:,1] + '_' + test[head_dam_XYZ_col].iloc[:,2] + '_' + test[head_dam_XYZ_col].iloc[:,3]
test['clean_purge_dam'] = test[head_dam_XYZ_col].iloc[:,4] + '_' + test[head_dam_XYZ_col].iloc[:,4] 
test['stage123_fill1'] = test[head_fill1_XYZ_col].iloc[:,1] + '_' + test[head_fill1_XYZ_col].iloc[:,2] + '_' + test[head_fill1_XYZ_col].iloc[:,3]
test['clean_purge_fill1'] = test[head_fill1_XYZ_col].iloc[:,4] + '_' + test[head_fill1_XYZ_col].iloc[:,5] 

test['stage123_fill2'] = test[head_fill2_XYZ_col].iloc[:,1] + '_' + test[head_fill2_XYZ_col].iloc[:,2] + '_' + test[head_fill2_XYZ_col].iloc[:,3]
test['clean_purge_fill2'] = test[head_fill2_XYZ_col].iloc[:,4] + '_' + test[head_fill2_XYZ_col].iloc[:,5] 


# 4. 상호작용항에 라벨 인코딩 적용
ordered_interaction_col_name = train_data.columns[train_data.columns.str.contains('dam|fill1|fill2')]

for i in ordered_interaction_col_name:

    interaction_le = LabelEncoder()
    train_data[i] = interaction_le.fit_transform(train_data[i])
    for label in list(test[i].unique()): #train에 없는 test값 처리
        if label not in list(interaction_le.classes_):
            interaction_le.classes_ = np.append(interaction_le.classes_,label)
    test[i] = interaction_le.transform(test[i])
    train_data[i] = train_data[i].astype("object") # 범주 변수 처리
    test[i] = test[i].astype("object") # 범주 변수 처리


# In[49]:


train_data.drop(columns=head_XYZ_col, inplace=True)
test.drop(columns=head_XYZ_col, inplace=True)


# In[50]:


# train_data['MEAN DISCHARGED TIME OF RESIN_Dam']=train_data[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']].mean(axis=1)
# train_data['STD DISCHARGED TIME OF RESIN_Dam']=train_data[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']].std(axis=1)
# train_data['SUM DISCHARGED TIME OF RESIN_Dam']=train_data[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']].sum(axis=1)

# train_data['MEAN Dispense Volume_Dam']=train_data[['Dispense Volume(Stage1) Collect Result_Dam',
#        'Dispense Volume(Stage2) Collect Result_Dam',
#        'Dispense Volume(Stage3) Collect Result_Dam']].mean(axis=1)
# train_data['STD Dispense Volume_Dam']=train_data[['Dispense Volume(Stage1) Collect Result_Dam',
#        'Dispense Volume(Stage2) Collect Result_Dam',
#        'Dispense Volume(Stage3) Collect Result_Dam']].std(axis=1)
# train_data['SUM Dispense Volume_Dam']=train_data[['Dispense Volume(Stage1) Collect Result_Dam',
#        'Dispense Volume(Stage2) Collect Result_Dam',
#        'Dispense Volume(Stage3) Collect Result_Dam']].sum(axis=1)

# #비율 - 같은 용량이 나오는 데에 걸리는 시간
# train_data['DIV Dispense Volume(Stage1)_Dam']=train_data['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam']/train_data['Dispense Volume(Stage1) Collect Result_Dam']
# train_data['DIV Dispense Volume(Stage2)_Dam']=train_data['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam']/train_data['Dispense Volume(Stage2) Collect Result_Dam']
# train_data['DIV Dispense Volume(Stage3)_Dam']=train_data['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']/train_data['Dispense Volume(Stage3) Collect Result_Dam']

# #Fill1
# #합계/평균/최대/최소/표준편차/분산/비율

# train_data['MEAN DISCHARGED TIME OF RESIN_Fill1']=train_data[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']].mean(axis=1)
# train_data['STD DISCHARGED TIME OF RESIN_Fill1']=train_data[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']].std(axis=1)
# train_data['SUM DISCHARGED TIME OF RESIN_Fill1']=train_data[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']].sum(axis=1)

# train_data['MEAN Dispense Volume_Fill1']=train_data[['Dispense Volume(Stage1) Collect Result_Fill1',
#        'Dispense Volume(Stage2) Collect Result_Fill1',
#        'Dispense Volume(Stage3) Collect Result_Fill1']].mean(axis=1)
# train_data['STD Dispense Volume_Fill1']=train_data[['Dispense Volume(Stage1) Collect Result_Fill1',
#        'Dispense Volume(Stage2) Collect Result_Fill1',
#        'Dispense Volume(Stage3) Collect Result_Fill1']].std(axis=1)
# train_data['SUM Dispense Volume_Fill1']=train_data[['Dispense Volume(Stage1) Collect Result_Fill1',
#        'Dispense Volume(Stage2) Collect Result_Fill1',
#        'Dispense Volume(Stage3) Collect Result_Fill1']].sum(axis=1)

# #비율 - 같은 용량이 나오는 데에 걸리는 시간
# # train_data['DIV Dispense Volume(Stage1)_Fill1']=train_data['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1']/train_data['Dispense Volume(Stage1) Collect Result_Fill1']
# # train_data['DIV Dispense Volume(Stage2)_Fill1']=train_data['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1']/train_data['Dispense Volume(Stage2) Collect Result_Fill1']
# # train_data['DIV Dispense Volume(Stage3)_Fill1']=train_data['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']/train_data['Dispense Volume(Stage3) Collect Result_Fill1']

# #Thickness(변화가 있는지도 비율로 보면 좋을듯)

# train_data['SUM THICKNESS_Dam']=train_data[['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam','THICKNESS 3 Collect Result_Dam']].sum(axis=1)
# train_data['MEAN THICKNESS_Dam']=train_data[['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam','THICKNESS 3 Collect Result_Dam']].mean(axis=1)
# train_data['STD THICKNESS_Dam']=train_data[['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam','THICKNESS 3 Collect Result_Dam']].std(axis=1)

# train_data['DIV1 THICKNESS_Dam']=train_data['THICKNESS 2 Collect Result_Dam']/train_data['THICKNESS 1 Collect Result_Dam']
# train_data['DIV2 THICKNESS_Dam']=train_data['THICKNESS 3 Collect Result_Dam']/train_data['THICKNESS 2 Collect Result_Dam']

# #1st/2nd/3rd - 'THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam''THICKNESS 3 Collect Result_Dam'
# train_data['SUM Pressure_AutoClave']=train_data[['1st Pressure Collect Result_AutoClave', '2nd Pressure Collect Result_AutoClave', '3rd Pressure Collect Result_AutoClave']].sum(axis=1)
# train_data['MEAN Pressure_AutoClave']=train_data[['1st Pressure Collect Result_AutoClave', '2nd Pressure Collect Result_AutoClave', '3rd Pressure Collect Result_AutoClave']].mean(axis=1)
# train_data['STD Pressure_AutoClave']=train_data[['1st Pressure Collect Result_AutoClave', '2nd Pressure Collect Result_AutoClave', '3rd Pressure Collect Result_AutoClave']].std(axis=1)

# train_data['DIV1 Pressure_AutoClave']=train_data['2nd Pressure Collect Result_AutoClave']/train_data['1st Pressure Collect Result_AutoClave']
# train_data['DIV2 Pressure_AutoClave']=train_data['3rd Pressure Collect Result_AutoClave']/train_data['2nd Pressure Collect Result_AutoClave']


# In[51]:


# test['MEAN DISCHARGED TIME OF RESIN_Dam']=test[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']].mean(axis=1)
# test['STD DISCHARGED TIME OF RESIN_Dam']=test[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']].std(axis=1)
# test['SUM DISCHARGED TIME OF RESIN_Dam']=test[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']].sum(axis=1)

# test['MEAN Dispense Volume_Dam']=test[['Dispense Volume(Stage1) Collect Result_Dam',
#        'Dispense Volume(Stage2) Collect Result_Dam',
#        'Dispense Volume(Stage3) Collect Result_Dam']].mean(axis=1)
# test['STD Dispense Volume_Dam']=test[['Dispense Volume(Stage1) Collect Result_Dam',
#        'Dispense Volume(Stage2) Collect Result_Dam',
#        'Dispense Volume(Stage3) Collect Result_Dam']].std(axis=1)
# test['SUM Dispense Volume_Dam']=test[['Dispense Volume(Stage1) Collect Result_Dam',
#        'Dispense Volume(Stage2) Collect Result_Dam',
#        'Dispense Volume(Stage3) Collect Result_Dam']].sum(axis=1)

# #비율 - 같은 용량이 나오는 데에 걸리는 시간
# test['DIV Dispense Volume(Stage1)_Dam']=test['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam']/test['Dispense Volume(Stage1) Collect Result_Dam']
# test['DIV Dispense Volume(Stage2)_Dam']=test['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam']/test['Dispense Volume(Stage2) Collect Result_Dam']
# test['DIV Dispense Volume(Stage3)_Dam']=test['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']/test['Dispense Volume(Stage3) Collect Result_Dam']

# #Fill1
# #합계/평균/최대/최소/표준편차/분산/비율

# test['MEAN DISCHARGED TIME OF RESIN_Fill1']=test[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']].mean(axis=1)
# test['STD DISCHARGED TIME OF RESIN_Fill1']=test[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']].std(axis=1)
# test['SUM DISCHARGED TIME OF RESIN_Fill1']=test[['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1',
#        'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']].sum(axis=1)

# test['MEAN Dispense Volume_Fill1']=test[['Dispense Volume(Stage1) Collect Result_Fill1',
#        'Dispense Volume(Stage2) Collect Result_Fill1',
#        'Dispense Volume(Stage3) Collect Result_Fill1']].mean(axis=1)
# test['STD Dispense Volume_Fill1']=test[['Dispense Volume(Stage1) Collect Result_Fill1',
#        'Dispense Volume(Stage2) Collect Result_Fill1',
#        'Dispense Volume(Stage3) Collect Result_Fill1']].std(axis=1)
# test['SUM Dispense Volume_Fill1']=test[['Dispense Volume(Stage1) Collect Result_Fill1',
#        'Dispense Volume(Stage2) Collect Result_Fill1',
#        'Dispense Volume(Stage3) Collect Result_Fill1']].sum(axis=1)

# #비율 - 같은 용량이 나오는 데에 걸리는 시간
# # test['DIV Dispense Volume(Stage1)_Fill1']=test['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1']/test['Dispense Volume(Stage1) Collect Result_Fill1']
# # test['DIV Dispense Volume(Stage2)_Fill1']=test['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1']/test['Dispense Volume(Stage2) Collect Result_Fill1']
# # test['DIV Dispense Volume(Stage3)_Fill1']=test['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']/test['Dispense Volume(Stage3) Collect Result_Fill1']

# #Thickness(변화가 있는지도 비율로 보면 좋을듯)

# test['SUM THICKNESS_Dam']=test[['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam','THICKNESS 3 Collect Result_Dam']].sum(axis=1)
# test['MEAN THICKNESS_Dam']=test[['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam','THICKNESS 3 Collect Result_Dam']].mean(axis=1)
# test['STD THICKNESS_Dam']=test[['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam','THICKNESS 3 Collect Result_Dam']].std(axis=1)

# test['DIV1 THICKNESS_Dam']=test['THICKNESS 2 Collect Result_Dam']/test['THICKNESS 1 Collect Result_Dam']
# test['DIV2 THICKNESS_Dam']=test['THICKNESS 3 Collect Result_Dam']/test['THICKNESS 2 Collect Result_Dam']

# #1st/2nd/3rd - 'THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam''THICKNESS 3 Collect Result_Dam'
# test['SUM Pressure_AutoClave']=test[['1st Pressure Collect Result_AutoClave', '2nd Pressure Collect Result_AutoClave', '3rd Pressure Collect Result_AutoClave']].sum(axis=1)
# test['MEAN Pressure_AutoClave']=test[['1st Pressure Collect Result_AutoClave', '2nd Pressure Collect Result_AutoClave', '3rd Pressure Collect Result_AutoClave']].mean(axis=1)
# test['STD Pressure_AutoClave']=test[['1st Pressure Collect Result_AutoClave', '2nd Pressure Collect Result_AutoClave', '3rd Pressure Collect Result_AutoClave']].std(axis=1)

# test['DIV1 Pressure_AutoClave']=test['2nd Pressure Collect Result_AutoClave']/test['1st Pressure Collect Result_AutoClave']
# test['DIV2 Pressure_AutoClave']=test['3rd Pressure Collect Result_AutoClave']/test['2nd Pressure Collect Result_AutoClave']


# In[52]:


train_data['DIV Dispense Volume(Stage2)_Dam']=train_data['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam']/train_data['Dispense Volume(Stage2) Collect Result_Dam']
train_data['DIV Dispense Volume(Stage3)_Dam']=train_data['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']/train_data['Dispense Volume(Stage3) Collect Result_Dam']

train_data['STD Dispense Volume_Fill1']=train_data[['Dispense Volume(Stage1) Collect Result_Fill1',
       'Dispense Volume(Stage2) Collect Result_Fill1',
       'Dispense Volume(Stage3) Collect Result_Fill1']].std(axis=1)


# In[53]:


test['DIV Dispense Volume(Stage2)_Dam']=test['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam']/test['Dispense Volume(Stage2) Collect Result_Dam']
test['DIV Dispense Volume(Stage3)_Dam']=test['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']/test['Dispense Volume(Stage3) Collect Result_Dam']

test['STD Dispense Volume_Fill1']=test[['Dispense Volume(Stage1) Collect Result_Fill1',
       'Dispense Volume(Stage2) Collect Result_Fill1',
       'Dispense Volume(Stage3) Collect Result_Fill1']].std(axis=1)


# In[ ]:





# In[54]:


#  'STD DISCHARGED TIME OF RESIN_Dam',
#  'STD Dispense Volume_Dam',
#  'DIV Dispense Volume(Stage1)_Dam',
#  'DIV Dispense Volume(Stage2)_Dam',
#  'DIV Dispense Volume(Stage3)_Dam',
#  'STD DISCHARGED TIME OF RESIN_Fill1',
#  'STD Dispense Volume_Fill1',
#  'DIV2 THICKNESS_Dam',
#  'STD Pressure_AutoClave',
#  'DIV1 Pressure_AutoClave',
#  'DIV2 Pressure_AutoClave'
    
    
    
    
# #      'MEAN Dispense Volume_Dam',
# #        'SUM Dispense Volume_Dam', 'DIV Dispense Volume(Stage2)_Dam',
# #        'DIV Dispense Volume(Stage3)_Dam', 'SUM DISCHARGED TIME OF RESIN_Fill1',
# #        'MEAN Dispense Volume_Fill1', 'STD Dispense Volume_Fill1',
# #        'DIV1 THICKNESS_Dam', 'DIV2 THICKNESS_Dam', 'SUM Pressure_AutoClave',
# #        'MEAN Pressure_AutoClave', 'DIV1 Pressure_AutoClave',
# #        'DIV2 Pressure_AutoClave'


#  'DIV Dispense Volume(Stage2)_Dam',
#  'DIV Dispense Volume(Stage3)_Dam',
#  'STD Dispense Volume_Fill1'


# In[55]:


#        'DIV1 THICKNESS_Dam', 'DIV2 THICKNESS_Dam', 'SUM Pressure_AutoClave',
#        'MEAN Pressure_AutoClave', 'DIV1 Pressure_AutoClave',
#        'DIV2 Pressure_AutoClave'


# In[56]:


train_data['Mismatch_Dam_Fill1'] = (train_data['Production Qty Collect Result_Fill1'] != train_data['Production Qty Collect Result_Dam']).astype(object)
train_data['Mismatch_Dam_Fill2'] = (train_data['Production Qty Collect Result_Fill2'] != train_data['Production Qty Collect Result_Dam']).astype(object)
train_data['Mismatch_Fill1_Fill2'] = (train_data['Production Qty Collect Result_Fill1'] != train_data['Production Qty Collect Result_Fill2']).astype(object)

test['Mismatch_Dam_Fill1'] = (test['Production Qty Collect Result_Fill1'] != test['Production Qty Collect Result_Dam']).astype(object)
test['Mismatch_Dam_Fill2'] = (test['Production Qty Collect Result_Fill2'] != test['Production Qty Collect Result_Dam']).astype(object)
test['Mismatch_Fill1_Fill2'] = (test['Production Qty Collect Result_Fill1'] != test['Production Qty Collect Result_Fill2']).astype(object)


# In[57]:


train_data['Mismatch_Receip_Dam_Fill1'] = (train_data['Receip No Collect Result_Fill1'] != train_data['Receip No Collect Result_Dam']).astype(object)
train_data['Mismatch_Receip_Dam_Fill2'] = (train_data['Receip No Collect Result_Fill2'] != train_data['Receip No Collect Result_Dam']).astype(object)
train_data['Mismatch_Receip_Fill1_Fill2'] = (train_data['Receip No Collect Result_Fill1'] != train_data['Receip No Collect Result_Fill2']).astype(object)

test['Mismatch_Receip_Dam_Fill1'] = (test['Receip No Collect Result_Fill1'] != test['Receip No Collect Result_Dam']).astype(object)
test['Mismatch_Receip_Dam_Fill2'] = (test['Receip No Collect Result_Fill2'] != test['Receip No Collect Result_Dam']).astype(object)
test['Mismatch_Receip_Fill1_Fill2'] = (test['Receip No Collect Result_Fill1'] != test['Receip No Collect Result_Fill2']).astype(object)


# In[58]:


train_data['Mismatch_equipment_Dam_Fill1'] = (train_data['Equipment_Fill1'] != train_data['Equipment_Dam']).astype(object)
train_data['Mismatch_equipment_Dam_Fill2'] = (train_data['Equipment_Fill2'] != train_data['Equipment_Dam']).astype(object)
train_data['Mismatch_equipment_Fill1_Fill2'] = (train_data['Equipment_Fill1'] != train_data['Equipment_Fill2']).astype(object)

test['Mismatch_equipment_Dam_Fill1'] = (test['Equipment_Fill1'] != test['Equipment_Dam']).astype(object)
test['Mismatch_equipment_Dam_Fill2'] = (test['Equipment_Fill2'] != test['Equipment_Dam']).astype(object)
test['Mismatch_equipment_Fill1_Fill2'] = (test['Equipment_Fill1'] != test['Equipment_Fill2']).astype(object)


# In[ ]:





# In[59]:


cols=['Stage1 Circle1 Distance Speed Collect Result_Dam',
 'Stage1 Circle2 Distance Speed Collect Result_Dam',
 'Stage1 Line1 Distance Speed Collect Result_Dam',
 'Stage1 Line2 Distance Speed Collect Result_Dam',
 'Stage1 Line4 Distance Speed Collect Result_Dam',
 'Stage2 Circle1 Distance Speed Collect Result_Dam',
 'Stage2 Circle2 Distance Speed Collect Result_Dam',
 'Stage2 Line2 Distance Speed Collect Result_Dam',
 'Stage2 Line3 Distance Speed Collect Result_Dam',
 'Stage2 Line4 Distance Speed Collect Result_Dam',
 'Stage3 Circle2 Distance Speed Collect Result_Dam',
 'Stage3 Line2 Distance Speed Collect Result_Dam',
 'Stage3 Line4 Distance Speed Collect Result_Dam',]


# In[60]:


train_data[cols]


# In[61]:


train_data


# In[62]:


drop_col=['Equipment_Dam','Equipment_Fill1','Equipment_Fill2',
          'Receip No Collect Result_Dam','Receip No Collect Result_Fill1','Receip No Collect Result_Fill2',
         'Production Qty Collect Result_Dam','Production Qty Collect Result_Fill1','Production Qty Collect Result_Fill2',
         ]
train=train_data.drop(columns=drop_col)
test=test.drop(columns=drop_col)


# In[63]:


# 먼저 각 컬럼을 datetime 형식으로 변환합니다.
train['Collect Date_Dam'] = pd.to_datetime(train['Collect Date_Dam'])
train['Collect Date_AutoClave'] = pd.to_datetime(train['Collect Date_AutoClave'])

# AutoClave에서 Dam을 뺀 시간을 계산하여 새로운 컬럼에 저장합니다.
train['Time_Difference_AutoClave_Dam'] = train['Collect Date_AutoClave'] - train['Collect Date_Dam']
train['Time_Difference_AutoClave_Dam'] = train['Time_Difference_AutoClave_Dam'].apply(lambda x: str(x).replace("0 days ", ""))
train['Time_Difference_AutoClave_Dam'] = train['Time_Difference_AutoClave_Dam'].astype("object")


# 결과 확인
print(train[['Collect Date_AutoClave', 'Collect Date_Dam', 'Time_Difference_AutoClave_Dam']].head())


# In[64]:


# 먼저 각 컬럼을 datetime 형식으로 변환합니다.
test['Collect Date_Dam'] = pd.to_datetime(test['Collect Date_Dam'])
test['Collect Date_AutoClave'] = pd.to_datetime(test['Collect Date_AutoClave'])

# AutoClave에서 Dam을 뺀 시간을 계산하여 새로운 컬럼에 저장합니다.
test['Time_Difference_AutoClave_Dam'] = test['Collect Date_AutoClave'] - test['Collect Date_Dam']

test['Time_Difference_AutoClave_Dam'] = test['Time_Difference_AutoClave_Dam'].apply(lambda x: str(x).replace("0 days ", ""))
test['Time_Difference_AutoClave_Dam'] = test['Time_Difference_AutoClave_Dam'].astype("object")
# 결과 확인
print(test[['Collect Date_AutoClave', 'Collect Date_Dam', 'Time_Difference_AutoClave_Dam']].head())


# In[65]:


train=train.drop(columns=['Collect Date_Dam', 'Collect Date_AutoClave', 'Collect Date_Fill1', 'Collect Date_Fill2'])
test=test.drop(columns=['Collect Date_Dam', 'Collect Date_AutoClave', 'Collect Date_Fill1', 'Collect Date_Fill2'])


# In[66]:


# time=train[['Collect Date_Dam', 'Collect Date_AutoClave', 'Collect Date_Fill1', 'Collect Date_Fill2']]
# time.loc[ train['Collect Date_Fill1']!= train['Collect Date_Fill2']]


# In[67]:


# time['Collect Date_AutoClave']-time['Collect Date_Dam']


# In[68]:


# # 먼저 각 컬럼을 datetime 형식으로 변환합니다.
# time['Collect Date_Dam'] = pd.to_datetime(time['Collect Date_Dam'])
# time['Collect Date_AutoClave'] = pd.to_datetime(time['Collect Date_AutoClave'])

# # AutoClave에서 Dam을 뺀 시간을 계산하여 새로운 컬럼에 저장합니다.
# time['Time_Difference_AutoClave_Dam'] = time['Collect Date_AutoClave'] - time['Collect Date_Dam']

# # 결과 확인
# print(time[['Collect Date_AutoClave', 'Collect Date_Dam', 'Time_Difference_AutoClave_Dam']].head())


# In[69]:


# time


# In[70]:


# # 먼저 각 컬럼을 datetime 형식으로 변환합니다.
# train_data['Collect Date_Dam'] = pd.to_datetime(train_data['Collect Date_Dam'])
# train_data['Collect Date_AutoClave'] = pd.to_datetime(train_data['Collect Date_AutoClave'])

# # AutoClave에서 Dam을 뺀 시간을 계산하여 새로운 컬럼에 저장합니다.
# train_data['Time_Difference_AutoClave_Dam'] = train_data['Collect Date_AutoClave'] - train_data['Collect Date_Dam']

# # 결과 확인
# print(train_data[['Collect Date_AutoClave', 'Collect Date_Dam', 'Time_Difference_AutoClave_Dam']].head())


# In[ ]:





# In[ ]:





# In[71]:


# 각 컬럼의 NA 개수 계산
na_counts = train_data.isna().sum()

# NA 개수가 0보다 큰 컬럼 목록 필터링
columns_with_na = na_counts[na_counts > 0].index.tolist()

# 결과 출력
print("NA 개수가 0보다 큰 컬럼들:")
columns_with_na


# In[72]:


# 각 컬럼의 NA 개수 계산
na_counts = test.isna().sum()

# NA 개수가 0보다 큰 컬럼 목록 필터링
columns_with_na = na_counts[na_counts > 0].index.tolist()

# 결과 출력
print("NA 개수가 0보다 큰 컬럼들:")
columns_with_na


# In[73]:


train.info(200)


# In[74]:


# train['Workorder1'] = train['Workorder'].str[:3]
# train['Workorder3']=train['Workorder'].apply(lambda x:x[5:8]).astype(int)

# test['Workorder1'] = test['Workorder'].str[:3]
# test['Workorder3']=test['Workorder'].apply(lambda x:x[5:8]).astype(int)

# train=train.drop(columns=['Workorder','Model.Suffix'])
# test=test.drop(columns=['Workorder','Model.Suffix'])


# In[75]:


data_type = train.dtypes.reset_index()
data_type.columns = ['col', 'type']
cat_features = list(data_type[data_type['type']=='object']['col'])
num_features = list(data_type[data_type['type']!='object']['col'])
data_type['unique'] = data_type['col'].apply(lambda x : train[x].nunique())
data_type['nullcount'] = data_type['col'].apply(lambda x : train[x].isnull().sum())
print('범주형 변수 수 : ' ,len(cat_features))
print('숫자형 변수 수 : ' ,len(num_features))


# In[76]:


train_num = train.loc[:,num_features]
original_cols = train_num.columns
rank = np.linalg.matrix_rank(train_num.values)
independent_cols = []
for i in range(len(original_cols)):
    sub_matrix = train_num[original_cols[:i+1]].values
    if np.linalg.matrix_rank(sub_matrix) > len(independent_cols):
        independent_cols.append(original_cols[i])
train = train.loc[:,cat_features + independent_cols]
test = test.loc[:,cat_features + independent_cols]


# In[77]:


data_type = train.dtypes.reset_index()
data_type.columns = ['col', 'type']
cat_features = list(data_type[data_type['type']=='object']['col'])
num_features = list(data_type[data_type['type']!='object']['col'])
data_type['unique'] = data_type['col'].apply(lambda x : train[x].nunique())
data_type['nullcount'] = data_type['col'].apply(lambda x : train[x].isnull().sum())
print('범주형 변수 수 : ' ,len(cat_features))
print('숫자형 변수 수 : ' ,len(num_features))


# In[ ]:





# In[78]:


num_features

# ['CURE END POSITION X Collect Result_Dam',
#  'CURE END POSITION Z Collect Result_Dam',
#  'CURE SPEED Collect Result_Dam',
#  'DISCHARGED SPEED OF RESIN Collect Result_Dam',
#  'DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam',
#  'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam',
#  'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam',
#  'Dispense Volume(Stage1) Collect Result_Dam',
#  'Dispense Volume(Stage2) Collect Result_Dam',
#  'Dispense Volume(Stage3) Collect Result_Dam',
#  'Machine Tact time Collect Result_Dam',
#  'PalletID Collect Result_Dam',
#  'Stage1 Circle1 Distance Speed Collect Result_Dam',
#  'Stage1 Circle2 Distance Speed Collect Result_Dam',
#  'Stage1 Line1 Distance Speed Collect Result_Dam',
#  'Stage1 Line2 Distance Speed Collect Result_Dam',
#  'Stage1 Line4 Distance Speed Collect Result_Dam',
#  'Stage2 Circle1 Distance Speed Collect Result_Dam',
#  'Stage2 Circle2 Distance Speed Collect Result_Dam',
#  'Stage2 Line2 Distance Speed Collect Result_Dam',
#  'Stage2 Line3 Distance Speed Collect Result_Dam',
#  'Stage2 Line4 Distance Speed Collect Result_Dam',
#  'Stage3 Circle2 Distance Speed Collect Result_Dam',
#  'Stage3 Line2 Distance Speed Collect Result_Dam',
#  'Stage3 Line4 Distance Speed Collect Result_Dam',
#  'THICKNESS 1 Collect Result_Dam',
#  'THICKNESS 2 Collect Result_Dam',
#  'THICKNESS 3 Collect Result_Dam',
#  'WorkMode Collect Result_Dam',
#  '1st Pressure Collect Result_AutoClave',
#  '1st Pressure 1st Pressure Unit Time_AutoClave',
#  '2nd Pressure Collect Result_AutoClave',
#  '2nd Pressure Unit Time_AutoClave',
#  '3rd Pressure Collect Result_AutoClave',
#  '3rd Pressure Unit Time_AutoClave',
#  'Chamber Temp. Collect Result_AutoClave',
#  'DISCHARGED SPEED OF RESIN Collect Result_Fill1',
#  'DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1',
#  'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1',
#  'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1',
#  'Dispense Volume(Stage1) Collect Result_Fill1',
#  'Dispense Volume(Stage2) Collect Result_Fill1',
#  'Dispense Volume(Stage3) Collect Result_Fill1',
#  'Machine Tact time Collect Result_Fill1',
#  'PalletID Collect Result_Fill1',
#  'WorkMode Collect Result_Fill1',
#  'CURE END POSITION X Collect Result_Fill2',
#  'CURE END POSITION Z Collect Result_Fill2',
#  'CURE SPEED Collect Result_Fill2',
#  'CURE STANDBY POSITION Z Collect Result_Fill2',
#  'Machine Tact time Collect Result_Fill2',
#  'PalletID Collect Result_Fill2',
#  'WorkMode Collect Result_Fill2',
#  'DIV Dispense Volume(Stage2)_Dam',
#  'DIV Dispense Volume(Stage3)_Dam',
#  'STD Dispense Volume_Fill1']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[79]:


# 'MEAN Dispense Volume_Dam',
#        'SUM Dispense Volume_Dam', 'DIV Dispense Volume(Stage2)_Dam',
#        'DIV Dispense Volume(Stage3)_Dam', 'SUM DISCHARGED TIME OF RESIN_Fill1',
#        'MEAN Dispense Volume_Fill1', 'STD Dispense Volume_Fill1',
#        'DIV1 THICKNESS_Dam', 'DIV2 THICKNESS_Dam', 'SUM Pressure_AutoClave',
#        'MEAN Pressure_AutoClave', 'DIV1 Pressure_AutoClave',
#        'DIV2 Pressure_AutoClave'


# In[ ]:





# In[80]:


for col in ['target']:
    le = LabelEncoder()
    le = le.fit(train[col])
    train[col] = le.transform(train[col])
    for label in list(test[col].unique()): #train에 없는 test값 처리
        if label not in list(le.classes_):
            le.classes_ = np.append(le.classes_,label)
    test[col] = le.transform(test[col])
cat_features.remove('target')


# In[81]:


target = train['target']
train = train.drop('target', axis=1)


# In[82]:


for col in cat_features:
  train[col] = train[col].astype("category")
  test[col] = test[col].astype("category")


# In[83]:


train['target'] = target.apply(lambda x : 1 if x==0 else 0)


# In[84]:


tuning_train, tuning_valid = train_test_split(train, test_size=0.2, random_state=0, stratify=train['target'])
# 적합에 필요한 데이터 분리
x_train = tuning_train.drop('target', axis=1)
y_train = tuning_train['target']
# 검증에 필요한 데이터 분리
x_valid = tuning_valid.drop('target', axis=1)
y_valid = tuning_valid['target']
# set seed
SEED = 0
# class weight
class_weights = compute_class_weight(classes=np.array([0,1]), y=tuning_train['target'], class_weight='balanced')
class_weights = {i: weight for i, weight in enumerate(class_weights)}
class_weights


# In[85]:


train


# In[86]:


cat_features


# In[87]:


space = {
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'n_estimators': 1000,
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
}

def f1_eval_metric(y_true, y_pred):
    # LightGBM은 예측 확률로 전달하므로, 이를 이진 예측으로 변환해야 합니다.
    y_pred_binary = np.round(y_pred)  # 0.5 기준으로 이진화
    f1 = f1_score(y_true, y_pred_binary)
    return 'f1_score', f1, True  

# 목적 함수 정의
def objective(params):
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    model = lgb.LGBMClassifier(**params, class_weight = class_weights, random_state=SEED, verbose=-1)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric=f1_eval_metric,
              categorical_feature = cat_features,
              callbacks = [lgb.early_stopping(stopping_rounds = 100), lgb.log_evaluation(period = 0)])

    preds = model.predict(x_valid)
    score = f1_score(y_valid, preds)

    # Hyperopt가 최소화를 수행하므로, 음수의 accuracy를 반환합니다.
    return {'loss': -score, 'status': STATUS_OK}


# In[88]:


# Trials 객체로 결과 기록
from hyperopt.early_stop import no_progress_loss
trials = Trials()
# 최적화 수행
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=300,
            trials=trials,
            rstate=np.random.default_rng(SEED),
            early_stop_fn=no_progress_loss(50))

# 최적 파라미터 출력
print("Best hyperparameters:", best)


# In[89]:


best['max_depth'] = int(best['max_depth'])
best['num_leaves'] = int(best['num_leaves'])
best['min_child_weight'] = int(best['min_child_weight'])
best['n_estimators'] = 1000
best


# In[ ]:


model = lgb.LGBMClassifier(**best, class_weight = class_weights, random_state=SEED, verbose=-1)
model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric=f1_eval_metric,
          categorical_feature = cat_features,
          callbacks = [lgb.early_stopping(stopping_rounds = 100), lgb.log_evaluation(period = 0)]
         )

preds = model.predict(x_valid)
score = f1_score(y_valid, preds)
print(score)


# In[ ]:


x_train = train.drop('target', axis=1)
y_train = train['target']


# In[ ]:


# class weight
SEED = 0
n_fold = 5
class_weights = compute_class_weight(classes=np.array([0, 1]), y=train['target'], class_weight='balanced')
class_weights = {i: weight for i, weight in enumerate(class_weights)}
class_weights


# In[ ]:


import pickle
from sklearn.metrics import roc_curve
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

model_scores = []  # 모델에서 뽑아내는 score
models = []
val_scores = []  # f1 score로 측정한 validation score
best_thresholds = []
pred_list = []

num = 0
cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)

# 폴더 경로 설정
model_save_path = "data/"  # 모델을 저장할 디렉토리

for train_idx, valid_idx in cv.split(x_train, y_train):
    num += 1
    scores = []
    print('\n', num, '번째 fold', "="*50)
    
    # train valid split
    x_train_cv, x_valid_cv, y_train_cv, y_valid_cv = x_train.iloc[train_idx], x_train.iloc[valid_idx], y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    # model fitting
    model = lgb.LGBMClassifier(**best, class_weight=class_weights, random_state=SEED, verbose=-1)
    model.fit(x_train_cv, y_train_cv, eval_set=[(x_valid_cv, y_valid_cv)],
              eval_metric=f1_eval_metric, categorical_feature=cat_features,
              callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)])
    
    models.append(model)

    # 모델을 파일로 저장
    with open(f"{model_save_path}model_fold_{num}.pkl", "wb") as file:
        pickle.dump(model, file)
    
    # validation evaluate
    val_proba = model.predict_proba(x_valid_cv)[:, 1]
    
    # ROC Curve 계산
    fpr, tpr, thresholds = roc_curve(y_valid_cv, val_proba)
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = [1 if y >= thresh else 0 for y in val_proba]
        f1 = f1_score(y_valid_cv, y_pred)
        f1_scores.append(f1)
    
    threshold = thresholds[np.argmax(f1_scores)]
    best_score = max(f1_scores)
    
    best_thresholds.append(threshold)
    val_scores.append(best_score)  # f1 score로 측정한 validation score
    
    # Final prediction
    pred = model.predict_proba(test.drop('target', axis=1))[:, 1]
    pred_list.append(pred)


# In[ ]:


best_thresholds


# In[ ]:


np.mean(best_thresholds)


# In[ ]:


final_pred = np.where(sum(pred_list)/n_fold >= np.mean(best_thresholds), 1, 0)
print('cv f1 mean score :', np.mean(val_scores))
print('test에서 1로 예측된 수:', sum(final_pred))


# In[ ]:


# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["target"] = final_pred
df_sub["target"] = df_sub["target"].apply(lambda x : 'AbNormal' if x==1 else 'Normal')
# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)


# In[ ]:


# from sklearn.inspection import permutation_importance
# import matplotlib.pyplot as plt
# all_importances = []

# # Compute permutation importance for each model on its corresponding validation set
# for model, (train_idx, valid_idx) in zip(models, cv.split(x_train, y_train)):
#     x_valid_cv, y_valid_cv = x_train.iloc[valid_idx], y_train.iloc[valid_idx]
    
#     # Permutation importance with F1 score as the evaluation metric
#     result = permutation_importance(model, x_valid_cv, y_valid_cv, scoring='f1', n_repeats=10, random_state=SEED)
#     all_importances.append(result.importances_mean)  # Take the mean importance across the repeats

# # Convert list to numpy array for easier averaging
# all_importances = np.array(all_importances)

# # Average the importances across all models
# mean_importances = np.mean(all_importances, axis=0)

# # Create a DataFrame for easier visualization
# feature_importance_df = pd.DataFrame({
#     'Feature': x_train_cv.columns,
#     'Importance': mean_importances
# })

# # Sort by importance and select top 30 features
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True).iloc[:30]

# # Visualize the top 30 feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
# plt.xlabel("Importance")
# plt.title("Feature Importance (Averaged Permutation Importance)")
# plt.gca().invert_yaxis()
# plt.show()


# In[ ]:


# # Create a DataFrame for easier visualization
# feature_importance_df = pd.DataFrame({
#     'Feature': x_train_cv.columns,
#     'Importance': mean_importances
# })



# # Sort by importance and select top 30 features
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True).iloc[-30:-1]

# # Visualize the top 30 feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
# plt.xlabel("Importance")
# plt.title("Feature Importance (Averaged Permutation Importance)")
# plt.gca().invert_yaxis()
# plt.show()


# In[ ]:


# feature_importance_df = pd.DataFrame({
#     'Feature': x_train_cv.columns,
#     'Importance': mean_importances
# })

# print(feature_importance_df.to_string())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 언더 샘플링
# 

# 데이타 불균형을 해결하기 위해 언더 샘플링을 진행합니다.
# 

# In[ ]:


# normal_ratio = 1.0  # 1.0 means 1:1 ratio

# df_normal = train_data[train_data["target"] == "Normal"]
# df_abnormal = train_data[train_data["target"] == "AbNormal"]

# num_normal = len(df_normal)
# num_abnormal = len(df_abnormal)
# print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}")

# df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=RANDOM_STATE)
# df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
# df_concat.value_counts("target")


# ### 데이터 분할
# 

# In[ ]:


# df_train, df_val = train_test_split(
#     df_concat,
#     test_size=0.3,
#     stratify=df_concat["target"],
#     random_state=RANDOM_STATE,
# )


# def print_stats(df: pd.DataFrame):
#     num_normal = len(df[df["target"] == "Normal"])
#     num_abnormal = len(df[df["target"] == "AbNormal"])

#     print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}" + f" ratio: {num_abnormal/num_normal}")


# # Print statistics
# print(f"  \tAbnormal\tNormal")
# print_stats(df_train)
# print_stats(df_val)


# ## 3. 모델 학습
# 

# ### 모델 정의
# 

# In[ ]:


# model = RandomForestClassifier(random_state=RANDOM_STATE)


# ### 모델 학습
# 

# In[ ]:


# features = []

# for col in df_train.columns:
#     try:
#         df_train[col] = df_train[col].astype(int)
#         features.append(col)
#     except:
#         continue

# train_x = df_train[features]
# train_y = df_train["target"]

# model.fit(train_x, train_y)


# ## 4. 제출하기
# 

# ### 테스트 데이터 예측
# 

# 테스트 데이터 불러오기
# 

# In[ ]:


# test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))


# In[ ]:


# df_test_x = test_data[features]

# for col in df_test_x.columns:
#     try:
#         df_test_x.loc[:, col] = df_test_x[col].astype(int)
#     except:
#         continue


# In[ ]:


# test_pred = model.predict(df_test_x)
# test_pred


# ### 제출 파일 작성
# 

# In[ ]:


# # 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
# df_sub = pd.read_csv("submission.csv")
# df_sub["target"] = test_pred

# # 제출 파일 저장
# df_sub.to_csv("submission.csv", index=False)


# **우측 상단의 제출 버튼을 클릭해 결과를 확인하세요**
# 

# In[ ]:




