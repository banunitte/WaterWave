# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

data1 = pd.DataFrame({'col1':range(0,11)})


data2 = pd.DataFrame({'col1':range(0,5),'col2':['babn','hgdggd','djhjdhj','djjdhd','djjdh']})


data2.rename(columns={'col2':'name'}, inplace=True)

result = pd.merge(data1,data2,on='col1',how='left')


concat = pd.concat([data1,data2],axis=1)

from sklearn.preprocessing import Imputer

######  here axis =0 means along column
imp = Imputer(missing_values='NaN',strategy='median',axis=0)
imp.fit(concat['col1'])


rows = []
import csv 

with open('top5.csv', 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    fields = next(csvreader)
  
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 
        print(row)
        print("++++++++++++")
  
    # get total number of rows 
qw = []
sam = [1,2,3,4,5,6,7,[3,4,5,6],5]
qw.append([2,3,4])
qw.extend([1,2,3,4])

from collections import defaultdict

di =  defaultdict(int)


with open('doc.txt','r') as red:
    for line in red:
        line = line.strip()
        for word in line.split():
            di[word] += 1
        print(line)

import re

pattern = '^a...s$'
test_string = 'abyss'
result = re.match(pattern, test_string)

if result:
  print("Search successful.")
else:
  print("Search unsuccessful.")	
  
  
import re

string = 'hello89 12 hib abbanu189 89. Howdy 34jjj89'
pattern4= '^91[0-9]{10}'
pattern3 = '\w{5}89$'
pattern1 = '[a-b]+banu[0-9]'
pattern2 = '[a-b]{2,4}banu[0-9]'
result = re.findall(pattern4, string) 
result = re.findall(pattern4, "jdjdd9199869611039") 
print(result)


list1 = ['physics', 'Biology', 'chemistry', 'maths']
list1.sort()
print ("list now : ", list1)

a = [1,56,2,3,23,45,4,5,6]

largest = 0
for i in a:
    if i > largest:
        largest = i
a.sort()
b=a.sort()
b = [2,3,4,19,20]
comman =[]
for i in b:
    if i in a:
        comman.append(i)
#for i in a:
#    for j in b:
#        if j in a:
#            
#            
#
#2 3 4
a = [1,2,3]
a[-3]

rev=0
n=4567

no= str(n)

no[::-1]
while n >0:
    rem = n%10
    rev = rev*10+rem
    n = n/10

def reverse_number(n):
    r = 0
    while n > 0:
        r *= 10
        r += n % 10
        n /= 10
    return r

print(reverse_number(123))



n = 123
for i in range(2,n):
    if n%i == 0:
        print(i)
        print("not a prime nuber")
        break
else:
    print("prime number")    

n=23
for i in range(n+1):
    for j in range(2,i):
        if i%j == 0:
            break
    else:
        print(str(i)+"is a prime number")
        
n1=0 
n2=1 
n=10
for i in range(2,n+1):
    n3=n1+n2
    print(n3)
    n1=n2
    n2=n3
    
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    