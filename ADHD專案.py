#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.decomposition import PCA

from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler


# ## 1. 資料清理

# In[2]:


# load data
ADHD_gender = pd.read_excel('01. ADHD詳細資料.xlsx', sheet_name=0).iloc[:,[0,4]] 
ADHD_open = pd.read_excel('01. ADHD詳細資料.xlsx', sheet_name=1)
ADHD_close = pd.read_excel('01. ADHD詳細資料.xlsx', sheet_name=2)
Control_gender = pd.read_excel('02.健康組詳細資料_更正版.xlsx', sheet_name=0).iloc[:,:2]
Control_open = pd.read_excel('02.健康組詳細資料_更正版.xlsx', sheet_name=1)
Control_close = pd.read_excel('02.健康組詳細資料_更正版.xlsx', sheet_name=2)

# 移除後續檢驗非過動症的人
ADHD_open = ADHD_open.drop([46,76,77])  # 等同drop(index=[46,76,77]) 
ADHD_close = ADHD_close.drop([46,76,77]) # 加 axis=1就是移除column 或寫成 columns=[]

# Remove missing values
ADHD_open = ADHD_open[ADHD_open['Fp1-D']!= "X"]
ADHD_close = ADHD_close[ADHD_close['Fp1-D']!= "X"]
Control_open = Control_open[Control_open['Fp1-D']!= "X"]
Control_close = Control_close[Control_close['Fp1-D']!= "X"]


# In[3]:


# combine (open & close)
ADHD_oc = pd.merge(ADHD_open, ADHD_close, how='inner', on='編號')
Control_oc = pd.merge(Control_open, Control_open, how='inner', on='編號')

# merge gender
ADHD = pd.merge(ADHD_oc, ADHD_gender, how='left', on='編號')
Control = pd.merge(Control_oc, Control_gender, how='left', on='編號')

# 增添target 0/1 代表有沒有患有ADHD
ADHD['tag'] = 1
Control['tag'] = 0


# In[4]:


# combine ADHD & control 
df = pd.concat([ADHD, Control]).reset_index(drop=True)

# remove 編號 欄位
df = df.drop('編號',axis=1)

# transform to int
df.iloc[:,:190] = df.iloc[:,:190].astype(float, errors = 'raise')

# 性別2為女性→改成0為女性
df['性別'].replace({2:0}, inplace=True)


# ## 2. 探索性分析

# ### 2.1 查看分布狀態

# In[5]:


df.describe()


# ### 2.2 Data imbalance check

# In[7]:


target = df.value_counts("tag")
print(target)

fig1, ax1 = plt.subplots()
ax1.pie(target, radius=1.5, labels=target.index, autopct='%1.1f%%')
ax1.axis('equal')
plt.title("Amount of ADHD", fontsize=14)
plt.show()


# ### 2.3 Categorical features

# In[5]:


Control = df[df['tag'] == 0]
ADHD = df[df['tag'] == 1]


A_target = ADHD['性別'].value_counts()
C_target = Control['性別'].value_counts()  

print(A_target)
print(C_target)

    
fig1, axs = plt.subplots(1, 2)
axs[0].pie(A_target, labels=A_target.index, autopct='%1.1f%%', shadow=None)
axs[0].axis('equal')
axs[0].set_title('ADHD Gender')
    
axs[1].pie(C_target, labels=C_target.index, autopct='%1.1f%%', shadow=None)
axs[1].axis('equal')
axs[1].set_title('Control Gender')
    
plt.show()
    


# ## 3. Model 架構

# ### 3.1  資料準備
# 

# In[6]:


# 劃分 特徵跟標籤
x = df.drop('tag',axis=1)
y = df['tag']


# In[7]:


# 定義 cross validation 的 各種 score 分數
def cv_score(pipe, x_train, y_train, cv_n):
    recall = cross_val_score(pipe, x_train, y_train, cv=cv_n, scoring='recall').mean()
    precision = cross_val_score(pipe, x_train, y_train, cv=cv_n, scoring='precision').mean()
    f1 = cross_val_score(pipe, x_train, y_train, cv=cv_n, scoring='f1').mean()
    accuracy = cross_val_score(pipe, x_train, y_train, cv=cv_n, scoring='accuracy').mean()

    print('Recall scores:', recall)
    print('Precision scores:', precision)
    print('F1 score:', f1)
    print('Accuracy:',accuracy)


# ### 3.2 Basic (Stratified K-fold )

# In[8]:


skf = StratifiedKFold(n_splits=5)


# In[9]:


print('LogisticRegression')
cv_score(LogisticRegression(random_state=42), x, y, skf)
print('')
print('-'*70)
print('')
print('KNN')
cv_score(KNeighborsClassifier(), x, y, skf)
print('')
print('-'*70)
print('')
print('LDA')
cv_score(LinearDiscriminantAnalysis(), x, y, skf)
print('')
print('-'*70)
print('')
print('SVM with rbf')
cv_score(SVC(kernel='rbf'), x, y, skf)
print('')
print('-'*70)
print('')
print('Random Forest')
cv_score(RandomForestClassifier(random_state=42), x, y, skf)
print('')
print('-'*70)
print('')
print('XGB')
cv_score(XGBClassifier(random_state=42), x, y, skf)


# ### 3.3 純 standardize + Stratified K-fold

# In[14]:


from sklearn.compose import ColumnTransformer # 用以整合欄位進行轉換

#categorical_cols = x_g.select_dtypes(include=["int64"]).columns.tolist()
numerical_cols = x.select_dtypes(include=["float64"]).columns.tolist()

# ColumnTransformer(transformers=[('欄位前贅字名稱/轉換器名稱', 轉換器 Transformer, [欲選取的欄位名稱])])
preprocessor = ColumnTransformer(
                    transformers=[('num', StandardScaler(), numerical_cols)],
                    remainder='passthrough')


# In[15]:


# standardize=True
lr_pipe = make_pipeline(preprocessor, LogisticRegression(random_state=42))
knn_pipe = make_pipeline(preprocessor, KNeighborsClassifier())
lda_pipe = make_pipeline(preprocessor, LinearDiscriminantAnalysis())
svm_pipe = make_pipeline(preprocessor, SVC(kernel='rbf'))
rf_pipe = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))
xgb_pipe = make_pipeline(preprocessor, XGBClassifier(random_state=42))


# In[18]:


print('LogisticRegression')
cv_score(lr_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('KNN')
cv_score(knn_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('LDA')
cv_score(lda_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('SVM with rbf')
cv_score(svm_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('Random Forest')
cv_score(rf_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('XGB')
cv_score(xgb_pipe, x, y, skf)


# ### 3.4 box-cox轉換 +  standardize + Stratified K-fold

# In[19]:


# ColumnTransformer(transformers=[('欄位前贅字名稱/轉換器名稱', 轉換器 Transformer, [欲選取的欄位名稱])])
preprocessor = ColumnTransformer(
                    transformers=[('num', PowerTransformer(method='box-cox'), numerical_cols)],
                    remainder='passthrough')


# In[20]:


# box cox 轉換的 pipeline (其中默認standardize=True)
lr_pipe = make_pipeline(preprocessor, LogisticRegression(random_state=42))
knn_pipe = make_pipeline(preprocessor, KNeighborsClassifier())
lda_pipe = make_pipeline(preprocessor, LinearDiscriminantAnalysis())
svm_pipe = make_pipeline(preprocessor, SVC(kernel='rbf'))
rf_pipe = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))
xgb_pipe = make_pipeline(preprocessor, XGBClassifier(random_state=42))


# In[21]:


print('LogisticRegression')
cv_score(lr_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('KNN')
cv_score(knn_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('LDA')
cv_score(lda_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('SVM with rbf')
cv_score(svm_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('Random Forest')
cv_score(rf_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('XGB')
cv_score(xgb_pipe, x, y, skf)


# ### 3.5 box-cox轉換 +  standardize + 特徵篩選 + Stratified K-fold

# In[25]:


from sklearn.feature_selection import  SelectPercentile, f_classif


# In[26]:


# SelectPercentile 用 f_classif(p-值) 篩選
lr_pipe = make_pipeline(preprocessor, SelectPercentile(f_classif, percentile= 90), LogisticRegression(random_state=42))
knn_pipe = make_pipeline(preprocessor, SelectPercentile(f_classif, percentile= 90), KNeighborsClassifier())
svm_pipe = make_pipeline(preprocessor, SelectPercentile(f_classif, percentile= 90), SVC(kernel='rbf'))
lda_pipe = make_pipeline(preprocessor, SelectPercentile(f_classif, percentile= 90), LinearDiscriminantAnalysis())
rf_pipe = make_pipeline(preprocessor, SelectPercentile(f_classif, percentile= 90), RandomForestClassifier(random_state=42))
xgb_pipe = make_pipeline(preprocessor, SelectPercentile(f_classif, percentile= 90), XGBClassifier(random_state=42))


# In[27]:


print('LogisticRegression')
cv_score(lr_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('KNN')
cv_score(knn_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('LDA')
cv_score(lda_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('SVM with rbf')
cv_score(svm_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('Random Forest')
cv_score(rf_pipe, x, y, skf)

print('')
print('-'*70)
print('')
print('XGB')
cv_score(xgb_pipe, x, y, skf)


# In[ ]:




