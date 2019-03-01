# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 23:51:15 2017

@author: Banuprakash
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os
os.chdir('D:\practice_file\poc')

chrun = pd.read_csv('Churn_Modelling.csv')

d = chrun['CreditScore'].std()
m = chrun['CreditScore'].mean()
def detect_outliers(data):
    
    if (m - 2 * d < data < m + 2 * d):
        return data
    else:
        return np.NaN
    
chrun['CreditScore'] = chrun['CreditScore'].apply(detect_outliers)  
chrun['CreditScore'].fillna(chrun['CreditScore'].mean(),inplace=True)


d = chrun['Age'].std()
m = chrun['Age'].mean()
chrun['Age'] = chrun['Age'].apply(detect_outliers) 
chrun['Age'].fillna(chrun['Age'].mean(),inplace=True) 

d = chrun['Tenure'].std()
m = chrun['Tenure'].mean()
chrun['Tenure'] = chrun['Tenure'].apply(detect_outliers) 
chrun['Tenure'].fillna(chrun['Tenure'].mean(),inplace=True) 

d = chrun['Balance'].std()
m = chrun['Balance'].mean()
chrun['Balance'] = chrun['Balance'].apply(detect_outliers) 
chrun['Balance'].fillna(chrun['Balance'].mean(),inplace=True) 


d = chrun['EstimatedSalary'].std()
m = chrun['EstimatedSalary'].mean()
chrun['EstimatedSalary'] = chrun['EstimatedSalary'].apply(detect_outliers) 
chrun['EstimatedSalary'].fillna(chrun['EstimatedSalary'].mean(),inplace=True)


dependent = chrun['Exited']

chrun.drop(['RowNumber','CustomerId','Surname','Exited'],axis=1,inplace=True)

Geography = pd.get_dummies(chrun['Geography'],drop_first=True)

Gender = pd.get_dummies(chrun['Gender'],drop_first=True)

cardholder = pd.get_dummies(chrun['HasCrCard'],drop_first=True)

activeMembers = pd.get_dummies(chrun['IsActiveMember'],drop_first=True)


products = pd.get_dummies(chrun['NumOfProducts'],drop_first=True)



chrun.drop(['Geography','Gender','HasCrCard','IsActiveMember','NumOfProducts'],axis=1,inplace=True)

chrun = pd.concat([Geography,Gender,cardholder,activeMembers,products,chrun],axis=1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(chrun, 
                                                       dependent, test_size=0.30, 
                                                        random_state=101)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
   
logmodel = LogisticRegression()
#classifier = SVC(kernel = 'rbf', random_state = 0)
rfe = RFE(logmodel,3)
fit = rfe.fit(X_train,y_train)
fit.n_features_
fit.support_
fit.ranking_

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#from sklearn.decomposition import PCA
#pca = PCA() #checked using  explains percentage of variance
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
## explained_variance -> explains percentage of variance
#explained_variance = pca.explained_variance_ratio_
#
#colnames = chrun.columns.tolist()