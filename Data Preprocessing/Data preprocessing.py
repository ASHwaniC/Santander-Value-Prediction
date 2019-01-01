import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#step 1: load train & test data, drop duplicate columns, round the features to NUM_OF_DECIMALS #decimals. here NUM_OF_DECIMALS is a experience value which can be tuned.

test_ID = test['ID']
y_train = train['target']
y_train = np.log1p(y_train)
train.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)
cols_with_onlyone_val = train.columns[train.nunique() == 1]
train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
test.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
NUM_OF_DECIMALS = 32
train = train.round(NUM_OF_DECIMALS)
test = test.round(NUM_OF_DECIMALS)
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i + 1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
train.drop(colsToRemove, axis=1, inplace=True) 
test.drop(colsToRemove, axis=1, inplace=True) 
train.shape

