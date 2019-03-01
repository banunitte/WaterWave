

import pandas as pd
import numpy as np
import seaborn as sns
import os


import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

os.chdir('D:\practice_file\poc')

train_df  = pd.read_csv('Churn_Modelling.csv')
sample_df = pd.read_csv('Churn_Modelling.csv')

sample_df.head().to_csv('top5.csv',index=False)

len(sample_df['Gender'].unique())



######

sam =sample_df[(sample_df.CreditScore >700) & (sample_df.Surname =='Yen')]
gender_dict = {'Female':0, 'Male':1}

def age_binning(age):
    # {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
    if age > 0 and age <= 17:
        return 0
    elif age > 18 and age <= 25:
        return 1
    elif age > 26 and age <= 35:
        return 2
    elif age > 36 and age <= 45:
        return 3
    elif age > 46 and age <= 50:
        return 4
    elif age > 51 and age <= 55:
        return 5
    else:
        return 6

train_df["Gender"] = train_df["Gender"].apply(lambda x: gender_dict[x])

train_df['Age'] = train_df['Age'].apply(age_binning)


def getCountVar(compute_df, count_df, var_name):
	grouped_df = count_df.groupby(var_name)
	count_dict = {}
	for name, group in grouped_df:
		count_dict[name] = group.shape[0]

	count_list = []
	for index, row in compute_df.iterrows():
		name = row[var_name]
		count_list.append(count_dict.get(name, 0))
	return count_list


print("Getting count features..")
train_df["Age_Count"] = getCountVar(train_df, train_df, "Age")
#test_df["Age_Count"] = getCountVar(test_df, train_df, "Age")


data.groupby(['month', 'item']).agg({'duration':sum,      # find the sum of the durations for each group
                                     'network_type': "count", # find the number of network type entries
                                     'date': 'first'})

sss=train_df.groupby('Geography')['CreditScore'].max()

qwe = sample_df.groupy('Geography')['CreditScore'].agg({'maximum':max,'Count':Count})

qwe1 = sample_df.groupby('Geography').agg({'CreditScore':['max','mean','count']})
###################3
means = sample_df.groupby('Geography')['Exited'].mean()
sample_df['Geography_TARGET_MEAN']=sample_df['Geography'].map(means)

sss['France']

def getPurchaseVar(compute_df, purchase_df, var_name):
        grouped_df = purchase_df.groupby(var_name)
        min_dict = {}
        max_dict = {}
        mean_dict = {}
        for name, group in grouped_df:
                min_dict[name] = min(np.array(group["CreditScore"]))
                max_dict[name] = max(np.array(group["CreditScore"]))
                mean_dict[name] = np.mean(np.array(group["CreditScore"]))
                # twentyfive_dict[name] = np.percentile(np.array(group["Purchase"]),25)
               # seventyfive_dict[name] = np.percentile(np.array(group["Purchase"]),75)

                
        print(min_dict)
        print(max_dict)
        print(mean_dict)
        min_list = []
        max_list = []
        mean_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                min_list.append(min_dict.get(name,0))
                max_list.append(max_dict.get(name,0))
                mean_list.append(mean_dict.get(name,0))

        return min_list, max_list, mean_list

min_price_list, max_price_list, mean_price_list = getPurchaseVar(train_df, train_df, "Geography")
djjd=pd.DataFrame({'min_dict':min_price_list})
train_df["User_ID_MinPrice"] = min_price_list
train_df["User_ID_MaxPrice"] = max_price_list
train_df["User_ID_MeanPrice"] = mean_price_list

train_y = np.array(train_df["Exited"])
train_df.drop(["Exited"], axis=1, inplace=True)

col = train_df.columns

cat_columns_list = ["User_ID", "Product_ID"]
for var in cat_columns_list:
                lb = LabelEncoder()
                #  full_var_data = pd.concat((train_df[var],test_df[var]),axis=0).astype('str')
                temp = lb.fit_transform(np.array(train_df))
              #  train_df[var] = lb.transform(np.array( train_df[var] ).astype('str'))
              #  test_df[var] = lb.transform(np.array( test_df[var] ).astype('str'))
              
sample_df.describe()