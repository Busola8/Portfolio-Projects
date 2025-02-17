# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:52:53 2025

@author: user
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
df_ = pd.read_csv("Year 2009-2010.csv", encoding='iso-8859-9')
df = df_.copy()
df.head()
df.isnull().sum()

df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T

df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]
cltv_df = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

cltv_df.columns = ['total_transaction', 'total_unit', 'total_price']
cltv_df.head()
#Average Order Value = Total Price / Total Transaction
cltv_df["average_order_value"] = cltv_df["total_price"] / cltv_df["total_transaction"]
cltv_df.head()

#Purchase Frequency = Total Transaction / Total Number of Customers

cltv_df.shape[0]

cltv_df["purchase_frequency"] = cltv_df["total_transaction"] / cltv_df.shape[0]
cltv_df.head()

# Churn Rate = 1 - Repeat Rate
# Repeat Rate = (Number of customers making multiple purchases) / (Total number of customers)
repeat_rate = cltv_df[cltv_df["total_transaction"] > 1].shape[0] / cltv_df.shape[0]
print(repeat_rate)

churn_rate = 1 - repeat_rate
print(churn_rate)

#Profit Margin = Total Price * 0.10
cltv_df['profit_margin'] = cltv_df['total_price'] * 0.10
cltv_df.head()

#Customer Value = Average Order Value * Purchase Frequency
cltv_df['customer_value'] = cltv_df['average_order_value'] * cltv_df["purchase_frequency"]
cltv_df.head()

#CLTV = (Customer Value / Churn Rate) * Profit Margin
cltv_df["cltv"] = (cltv_df["customer_value"] / churn_rate) * cltv_df["profit_margin"]
cltv_df.sort_values(by="cltv", ascending=False).head()

#creating segments
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.sort_values(by="cltv", ascending=False).head()

cltv_df.groupby("segment").agg({"cltv": ["mean", "sum"]})
cltv_df.to_csv("cltc_c.csv")


