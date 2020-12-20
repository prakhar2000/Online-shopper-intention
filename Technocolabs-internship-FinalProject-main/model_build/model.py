# -*- coding: utf-8 -*-
"""model.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

df=pd.read_csv("online_shoppers_intention.csv")
df.head(5)

df.info()

df.nunique()

df.VisitorType.unique()
df.PageValues.unique()

df.drop(columns=["Region","TrafficType","OperatingSystems","Administrative_Duration","Informational_Duration","Browser","BounceRates","ExitRates"],axis=1,inplace=True)

df.head()

df["Revenue"]=df["Revenue"].astype(int) 
df["Weekend"]=df["Weekend"].astype(int) 
df.head()

df = pd.get_dummies(df, columns=[ "Month","VisitorType"], drop_first=True)
df.head()

revenue_column=df['Revenue']
df.drop("Revenue",inplace=True,axis=1)

X = df.iloc[:, :].values
y = revenue_column.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#pip install catboost

from catboost import CatBoostClassifier

clf = CatBoostClassifier(
    iterations=500, 
    learning_rate=0.1, 
    depth=4,
    l2_leaf_reg=5,
    loss_function='CrossEntropy'
)
clf.fit(X_train, y_train,  
        eval_set=(X_test, y_test), 

);

clf.score(X_test,y_test)

import pickle
pickle.dump(clf, open('shoppers.pkl', 'wb'))
