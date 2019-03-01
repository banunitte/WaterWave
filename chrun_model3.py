# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 23:37:34 2017

@author: Banuprakash
"""

from sklearn.feature_selection import RFE

import pandas as pd
import numpy as np
import seaborn as sns
import os
os.chdir('D:\practice_file\poc')

chrun = pd.read_csv('Churn_Modelling.csv')

dependent = chrun['Exited']

chrun.drop(['RowNumber','CustomerId','Surname','Exited'],axis=1,inplace=True)

Geography = pd.get_dummies(chrun['Geography'],drop_first=True)

Gender = pd.get_dummies(chrun['Gender'],drop_first=True)

cardholder = pd.get_dummies(chrun['HasCrCard'],drop_first=True)

activeMembers = pd.get_dummies(chrun['IsActiveMember'],drop_first=True)

tenure = pd.get_dummies(chrun['Tenure'],drop_first=True)

products = pd.get_dummies(chrun['NumOfProducts'],drop_first=True)

AGe = activeMembers = pd.get_dummies(chrun['Age'],drop_first=True)

chrun.drop(['Geography','Age','Gender','HasCrCard','IsActiveMember',
            'Tenure','NumOfProducts'],axis=1,inplace=True)

chrun = pd.concat([chrun,Geography,Gender,cardholder,activeMembers,AGe,tenure,products],axis=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()






from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(chrun, 
                                                       dependent, test_size=0.30, 
                                                        random_state=101)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
   
logmodel = LogisticRegression(random_state = 100)
#################################################333
X_train = X_train[:,list1]
X_test = X_test[:,list1]

logmodel.fit(X_train,y_train)

y_pred = logmodel.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


############################
rfe=RFE(logmodel,5)

fit=rfe.fit(X_train,y_train)

first =  fit.n_features_
sec = fit.support_
third = fit.ranking_

print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

list1 = []
for i,j in enumerate(third):
    if j == 1:
        list.append(i)
        
        
i =-1 
for m in third:
     i = i+1
     if m == 1:
        list1.append(i)
        
X_train = X_train[:,list1]        
       
        
    


