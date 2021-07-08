#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:21:45 2021

@author: delizhu
"""

# To get closing price data 
from pandas_datareader import data as pdr 
import yfinance as yf 
yf.pdr_override() 
 
# Plotting graphs 
import matplotlib.pyplot as plt 
import seaborn 
 
# Data manipulation 
import numpy as np 
import pandas as pd

stock = pdr.get_data_yahoo("^NSEI",start = '2015-01-03', end = '2019-12-31')
stock['high'] = stock.Close.shift(1).rolling(window=5).max()
stock['low'] = stock.Close.shift(1).rolling(window=5).min()
stock['avg'] = stock.Close.shift(1).rolling(window=5).mean()
#进场信号
stock['long_entry'] = stock.Close > stock.high
stock['short_entry'] = stock.Close < stock.low
#出场信号
stock['long_exit'] = stock.Close < stock.avg
stock['short_exit'] = stock.Close > stock.avg
#position统一信号
stock['positions_long'] = np.nan 
stock.loc[stock.long_entry,'positions_long']= 1 
stock.loc[stock.long_exit,'positions_long']= 0 

stock['positions_short'] = np.nan 
stock.loc[stock.short_entry,'positions_short']= -1 
stock.loc[stock.short_exit,'positions_short']= 0 
#stock = stock.fillna(0)

stock['Signal'] = stock.positions_long + stock.positions_short 
stock = stock.fillna(0)

stock['Daily_returns'] = np.log(stock['Close']/stock['Close'].shift(1))
Cumulative_Stock_returns = np.cumsum(stock['Daily_returns'])

stock['Startegy_returns'] = stock['Daily_returns']* stock['Signal'].shift(1)
Cumulative_Strategy_returns = np.cumsum(stock['Startegy_returns'])

plt.figure(figsize=(10,5))
plt.plot(Cumulative_Stock_returns, color='r',label = 'Stock Returns')
plt.plot(Cumulative_Strategy_returns, color='g', label = 'Turtle Returns')
plt.legend()
plt.show()
