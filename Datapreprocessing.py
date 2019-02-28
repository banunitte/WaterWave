# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:41:42 2017

@author: Banuprakash
"""
import os
os.chdir('D:\practice_file\pre-processing')
import numpy as np
import pandas as pd
import seaborn as sns

df= pd.read_csv('adult.csv',na_values=['#NAME?'])
                                       
df.info()   
####1ST ALWAYS LOOK INTO DATAFRAME INFORMATION
df['income']=[0 if x == '<=50K' else 1 for x in df['income']]        

X = df.drop('income',axis=1)      
Y = df['income']                      
                                      

#####deciding which catogorical variable to take into acount when there are lots of unique categories
dummy_list =[]
for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        dummy_list.append(col_name)
        unique_category = len(X[col_name].unique())
        print("feature '{col_name}' has ;{unique_category}'".format(col_name=col_name,unique_category=unique_category))
        
lis_ofdummies = X.columns.tolist()        

print(X['native.country'].value_counts().sort_values(ascending=False).head())

X['native.country'] = [x if x == 'United-States' else 'Others' for x in X['native.country']]


def create_dummy(df,to_dummy):
    for x in to_dummy:
        dummies = pd.get_dummies(df[x],prefix=x,dummy_na=False)
        df=df.drop(x,axis=1)
        df=pd.concat([df,dummies],axis=1)
    return df    

X = create_dummy(X,dummy_list[0:8])


###checking how much of your data is missing
X.isnull().sum().sort_values(ascending=False).head()


sns.heatmap(X['fnlwgt'].isnull(),yticklabels=False,cbar=False,cmap='viridis')


from sklearn.preprocessing import Imputer

######  here axis =0 means along column
imp = Imputer(missing_values='NaN',strategy='median',axis=0)
imp.fit(X)


def find_outliers(x):
    q1 = np.percentile(x,25)
    q3 = np.percentile(x,75)
    iqr = q3-q1
    floor = q1 - 1.5*iqr
    ceiling = q3 +1.5*iqr
    outlier_indices = list(x.index[(x < floor) | (x >ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices,outlier_values        



tukey_indices,tukey_values = find_outliers(X['age'])
print(np.sort(tukey_values))

##############################
X['age'][tukey_indices] = X['age'].mean()
#################################


from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate


def find_outiers_kde(x):
    x_scaled = scale(list(map(float,x)))
    kde = KDEUnivariate(x_scaled)
    kde.fit(bw="scott",fft=True)
    pred = kde.evaluate(x_scaled)
    
    n = sum(pred < 0.5)
    outlierindices=np.asarray(pred).argsort()[:n]
    outliervalue=np.asarray(x)[outlierindices]
    return outlierindices,outliervalue

kde_indices,kde_value = find_outiers_kde(X['age'])



##################FEATURE ENGINEERING##################
##USE POLINOMIAL FEATURE IN SKLEARN.PREPOROCESSING  TO CREAT  TWO WAY INTERACTIION FOR
#ALL FEATURES

from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

def add_interaction(df):
    ##get feature name
    combos = list(combinations(list(df.columns),2))
    colnames = list(df.columns) + ['_'.join(x) for x in  combos]
    
    
    ##find ineractions
    poly = PolynomialFeatures(interaction_only=True,include_bias=False)
    df=poly.fit_transform(df)
    df=pd.DataFrame(df)
    df.columns = colnames
    
    
    #remove interaction terms with all 0 values
    noint_indices = [i for i,x in enumerate(list((df == 0).all())) if x]
    df =df.drop(df.columns[noint_indices],axis=1)
    
    return df
X =  add_interaction(X)
print(X.head(5))
###pca#################

from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
X_pca = pd.DataFrame(pca.fit_transform(X))
#################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)

import sklearn.feature_selection
select = sklearn.feature_selection.SelectKBest(k=10)
selected_feature = select.fit(X_train,y_train)

##########################
feature_score = selected_feature.scores_
maxacore = feature_score.max
scores= selected_feature.scores_

##########################
indices_selected = selected_feature.get_support(indices=True)
columns_selected = [X.columns[i] for i in indices_selected]





X_train_selected = X_train[columns_selected]
X_test_selected = X_test[columns_selected]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def find_model_perf(X_train,y_train,X_test,y_test):
    model =  LogisticRegression()
    model.fit(X_train,y_train)
    y_hat = [x[1] for x in  model.predict_proba(X_test)]
    auc = roc_auc_score(y_test,y_hat)
    return auc


accuracy = find_model_perf(X_train,y_train,X_test,y_test)