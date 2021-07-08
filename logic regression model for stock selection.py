#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:00:23 2021

@author: delizhu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Technical Indicators
#import talib as ta
#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score#cross validation
#Data fetching 
import yfinance as yf
from pandas_datareader import data as pdr
#import statsmodels.api as sm
yf.pdr_override()

df = pdr.get_data_yahoo("^NSEI",start = '2015-01-03', end = '2019-12-31')
df = df.iloc[:,:6]
df = df.dropna()
Return = (df['Adj Close'].shift(-1)/df['Adj Close'] - 1) * 100 # 每日收盘价百分比变化
df['S_10'] = df['Close'].rolling(window=10).mean()#10日移动均线
df['Cor'] = df['Close'].rolling(window =10).corr(df['S_10'])#与10日均值相关性
#df['RSI'] = ta.RSI(np.array(df['Close']),timeperiod = 10)
df['Open-Close'] = df['Open'] - df['Close'].shift(1)#今日开盘与昨日封盘价差
df['Open-Open'] = df['Open'] - df['Open'].shift(1)#今日开盘与昨日开盘价差
df['Volume'] = df['Volume'].shift(1).values/10000 #前一日成交量也作为指标
df = df.dropna()

#df = df.rename('Today')
#df = df.reset_index()

#for i in range(1,6):
#    df['Lag '+str(i)] = df['Today'].shift(i)
    
#df['Volume'] = df.Volume.shift(1).values/1000000 #前一日成交量也作为指标
#df = df.dropna()
#df['Direction'] = [1 if i > 0 else 0 for i in df['Today']]

#df = sm.add_constant(df)
X = df.iloc[:,:9]
y = np.where(df['Close'].shift(-1) > df['Close'],1,0)
#split the data into training set and test set
split = int(len(df)*0.7)
x_train,x_test,y_train,y_test = X[:split],X[split:],y[:split],y[split:]



model = LogisticRegression()
model = model.fit(x_train,y_train)
probability = model.predict_proba(x_test)
prediction = model.predict(x_test)#
#confusion_matrix = metrics.confusion_matrix(y_test,prediction)



def confusion_matrix(act,pred):
    predtrans = ['Up' if i > 0.5 else 'Down' for i in pred ]
    actuals  = ['Up' if i >0 else 'Down' for i in act ]
    confusion_matrix = pd.crosstab(pd.Series(actuals),
                                   pd.Series(predtrans),
                                   rownames = ['Actual'],
                                   colnames = ['Predicted'])
    return confusion_matrix

confusion_matrix = confusion_matrix(y_test,prediction)
pred_result = metrics.classification_report(y_test, prediction)
accuracy = model.score(x_test,y_test)
## Cross validaiton
cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print('--------------------')
print('Cross Validation:',cross_val)
print('--------------------')
print('Mean of CV:',cross_val.mean())

## Trading strategy
df['Predicted_Signal'] = model.predict(X)
df['Nifty_returns'] = np.log(df['Close']/df['Close'].shift(1))
Cumulative_Nifty_returns = np.cumsum(df[split:]['Nifty_returns'])#累积收益率

df['Startegy_returns'] = df['Nifty_returns']* df['Predicted_Signal'].shift(1)
Cumulative_Strategy_returns = np.cumsum(df[split:]['Startegy_returns'])



plt.figure(figsize=(10,5))
plt.plot(Cumulative_Nifty_returns, color='r',label = 'Nifty Returns')
plt.plot(Cumulative_Strategy_returns, color='g', label = 'Strategy Returns')
plt.legend()
plt.show()












