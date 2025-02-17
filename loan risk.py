# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:30:32 2025

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, auc
from imblearn.over_sampling import SMOTE, RandomOverSampler
from scipy.stats import skew

data = pd.read_csv("credit_risk_dataset.csv")
correl = data.corr(numeric_only = True)
unique = data.nunique()
data["cb_person_default_on_file"] = data["cb_person_default_on_file"].replace({"Y":1,"N":0})

corellation = data.corr()
data.info()

plt.figure(figsize = (15, 10))
# container = plt.bar(x = data["cb_person_default_on_file"], height = data["person_age"], width = 0.5, color = "brown")
# plt.bar_label(container = container, padding = 5)
plt.scatter(x = data["person_home_ownership"],y = data["person_age"], c = "brown")
plt.ylabel("person_age")
plt.xlabel("person_home_ownership")
plt.show()

plt.figure(figsize=(12, 8))
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap for Numeric Variables")
plt.show()

# Pairplot for numeric features and loan status
sns.pairplot(data, vars=["person_income", "loan_amnt", "loan_int_rate", "loan_percent_income"], hue="loan_status", diag_kind="kde", palette="Set2")
plt.suptitle("Pairplot of Numeric Features by Loan Status", y=1.02)
plt.show()

# Boxplot: Loan Amount by Loan Intent-no
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="loan_intent", y="loan_amnt", hue="loan_status", palette="Set3")
plt.title("Loan Amount Distribution by Loan Intent and Loan Status")
plt.xticks(rotation=45)
plt.show()

# Countplot: Loan Grade Distribution by Loan Status
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="loan_grade", hue="loan_status", palette="Set1")
plt.title("Loan Grade Distribution by Loan Status")
plt.show()

# Barplot: Average Interest Rate by Home Ownership and Loan Status
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x="person_home_ownership", y="loan_int_rate", hue="loan_status", ci=None, palette="Set2")
plt.title("Average Interest Rate by Home Ownership and Loan Status")
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(data=data, x="person_home_ownership",  y="cb_person_default_on_file", hue="loan_status", ci=None, palette="Set2")
plt.title("Home Ownership and Loan Status comparison")
plt.show()

# Plot: Home Ownership vs Loan Status
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="person_home_ownership", hue="loan_status", palette="Set2")
plt.title("Home Ownership vs Loan Status")
plt.xlabel("Home Ownership")
plt.ylabel("Count")
plt.show()

# Filter data for those who took loans
loan_taken = data[data["loan_status"] == 1]

# Plot: Distribution of Home Ownership for Those Who Took Loans
plt.figure(figsize=(10, 6))
sns.countplot(data=loan_taken, x="person_home_ownership", palette="coolwarm")
plt.title("Distribution of Home Ownership for Those Who Took Loans")
plt.xlabel("Home Ownership")
plt.ylabel("Count")
plt.show()


# Plot: Loan Intent Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="loan_intent", palette="viridis", order=data["loan_intent"].value_counts().index)
plt.title("Loan Intent Distribution")
plt.xlabel("Loan Intent")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Plot: Age vs Home Ownership
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="person_home_ownership", y="person_age", palette="Set3")
plt.title("Age Distribution by Home Ownership")
plt.xlabel("Home Ownership")
plt.ylabel("Age")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=data, x="person_home_ownership", y="person_age", palette="Set1", estimator=np.median)
plt.title("Median Age by Home Ownership")
plt.xlabel("Home Ownership")
plt.ylabel("Median Age")
plt.show()

data.hist()
original_median = data['loan_int_rate'].median()
data['loan_int_rate'] = data['loan_int_rate'].fillna(original_median)
imputed_median = data['loan_int_rate'].median()  #Compare the Imputed Median with the Original Median
bias = (imputed_median - original_median) / original_median * 100
print(f"Imputation bias percentage: {bias}%")

data.dropna(subset=['person_emp_length'], inplace=True)


print(skew(data['loan_int_rate']))

data= pd.get_dummies(data, columns = ["person_home_ownership","loan_grade","loan_intent"], drop_first = True)

# Prepare the dataset
x = data.drop("loan_status", axis=1)
y = data["loan_status"]

# Scale the feature variables
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(x)
scaled_dataset = pd.DataFrame(scaled_dataset, columns=x.columns)

scaled_data_reset = scaled_dataset.reset_index(drop=True)
y_reset = y.reset_index(drop=True)
# Remove outliers using Z-score thresholds (-3 to 3 for scaled data)
removed_outliers_dataset = scaled_dataset[(scaled_dataset < 3).all(axis=1) & (scaled_dataset > -3).all(axis=1)]

y_filtered = y.iloc[removed_outliers_dataset.index]
print(removed_outliers_dataset.shape)
print(y_filtered.shape)
x_train, x_test, y_train, y_test = train_test_split(removed_outliers_dataset, y_filtered, test_size=0.2, random_state=0)

# Train-test split
#x_train, x_test, y_train, y_test = train_test_split(removed_outliers_dataset, y, test_size=0.2, random_state=0)
mising = removed_outliers_dataset.isnull().sum()
#Handle class imbalance using SMOTE 
smote = SMOTE(random_state=0)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Train the XGBoost model
classifier = XGBClassifier(random_state=0)
model = classifier.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)

# Training metrics
train_confusion_matrix = confusion_matrix(y_train, y_pred)
train_classification_report = classification_report(y_train, y_pred)
train_accuracy = accuracy_score(y_train, y_pred)
train_precision = precision_score(y_train, y_pred)
train_recall = recall_score(y_train, y_pred)
train_f1_score = f1_score(y_train, y_pred)

# Testing metrics
test_confusion_matrix = confusion_matrix(y_test, y_pred1)
test_classification_report = classification_report(y_test, y_pred1)
test_accuracy = accuracy_score(y_test, y_pred1)
test_precision = precision_score(y_test, y_pred1)
test_recall = recall_score(y_test, y_pred1)
test_f1_score = f1_score(y_test, y_pred1)

# Print Results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

















