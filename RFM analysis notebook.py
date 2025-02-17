# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:24:49 2025

@author: user
"""

import numpy as np
import pandas as pd
import datetime as dt

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Year 2009-2010.csv",encoding='iso-8859-9')
df = data.copy()
brief = df.head()
missing = df.isnull().sum()
info = df.info()
description = df.describe()
df.hist()
correlation = df.corr(numeric_only = True)
duplicates = df.duplicated()
unique = df.nunique()
df["Description"].value_counts().head()

#total quantity of an individual unique product sold
df.groupby("Description")["Quantity"].sum().head()
#most sold
df.groupby("Description")["Quantity"].sum().sort_values(ascending = False).head()

df["TotalPrice"] = df["Quantity"] * df["Price"]

#Price cannot have negative values. This is presence of returns
df = df[df["Quantity"]>0]
df.dropna(inplace=True)
show = df.describe()

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
last_purchase_date = df.groupby("Customer ID")["InvoiceDate"].max()
todays_date = df["InvoiceDate"].max()+pd.Timedelta(days = 2)
recency = (todays_date - last_purchase_date).dt.days

frequency = df.groupby("Customer ID")["Invoice"].nunique()
monetary = df.groupby("Customer ID")["TotalPrice"].sum()

rfm = pd.DataFrame({
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary
})

rfm = rfm[rfm["Monetary"] > 0] # the minimum value of monetary is 0, and this is not desired.

rfm["recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


rfm[rfm["RFM_SCORE"] == "55"].head()
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

#r'[1-2][1-2]', r'[1-2][3-4]', r'[1-2]5: These are regular expressions.  
#r'[1-2][1-2]' represents customers with a "Recency" score of 1 or 2 and a "Frequency" score of 1 or 2.

#transform RFM scores in the "RFM_SCORE" column into the segment names
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "cant_loose"].head()

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)
new_df.to_csv("new_customers.csv")
rfm.to_csv("rfm.csv")














