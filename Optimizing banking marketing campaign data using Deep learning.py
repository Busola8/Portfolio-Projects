# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:30:30 2025

@author: user
"""


import tensorflow
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.optimizers import RMSprop
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
dataset = pd.read_csv("bank.csv")
df = dataset.copy()
df.head()

df.deposit.value_counts(normalize = True)

sns.set_style('darkgrid')
plt.figure(figsize=(7, 6))
sns.countplot(x= df.deposit)
plt.title('Target Variable Distribution')
plt.show()

print(df.month.value_counts(), "\n")
print(df.job.value_counts())

labelencoder = LabelEncoder()
df["job"] = labelencoder.fit_transform(df["job"])
labelencoder = LabelEncoder()
df["marital"] = labelencoder.fit_transform(df["marital"])
labelencoder = LabelEncoder()
df["education"] = labelencoder.fit_transform(df["education"])
labelencoder = LabelEncoder()
df["default"] = labelencoder.fit_transform(df["default"])
labelencoder = LabelEncoder()
df["housing"] = labelencoder.fit_transform(df["housing"])
labelencoder = LabelEncoder()
df["loan"] = labelencoder.fit_transform(df["loan"])
labelencoder = LabelEncoder()
df["contact"] = labelencoder.fit_transform(df["contact"])
labelencoder = LabelEncoder()
df["month"] = labelencoder.fit_transform(df["month"])
labelencoder = LabelEncoder()
df["poutcome"] = labelencoder.fit_transform(df["poutcome"])
labelencoder = LabelEncoder()
df["deposit"] = labelencoder.fit_transform(df["deposit"])


def plot_correlation_map(df):

    corr = df.corr()

    s, ax = plt.subplots(figsize=(15, 20))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    s = sns.heatmap(corr,
                    cmap=cmap,
                    square=True,
                    cbar_kws={'shrink': .9},
                    ax=ax,
                    annot=True,
                    annot_kws={'fontsize': 12})


plot_correlation_map(df)

from itertools import cycle
import plotly.graph_objects as go

palette = cycle(["#ffd670", "#70d6ff", "#ff4d6d", "#8338ec", "#90cf8e"])
targ = df.corrwith(df['deposit'], axis=0)
val = [str(round(v, 1) * 100) + '%' for v in targ.values]
fig = go.Figure()
fig.add_trace(
    go.Bar(y=targ.index,
            x=targ.values,
            orientation='h',
            text=val,
            marker_color=next(palette)))
fig.update_layout(title="Correlation of variables with Target",
                  width=1000,
                  height=500,
                  paper_bgcolor='rgb(0,0,0,0)',
                  plot_bgcolor='rgb(0,0,0,0)')
#Preprocessing
deposit_yes = df[df['deposit'] == 'yes']
deposit_no = df[df['deposit'] == 'no']
df.head()

fig, ax = plt.subplots(2, 2, figsize=(12,10))

b1 = ax[0, 0].bar(deposit_yes['day'].unique(),height = deposit_yes['day'].value_counts(),color='#000000')
b2 = ax[0, 0].bar(deposit_no['day'].unique(),height = deposit_no['day'].value_counts(),bottom = deposit_yes['day'].value_counts(),color = '#DC4405') 
ax[0, 0].title.set_text('Day of week')
#ax[0, 0].legend((b1[0], b2[0]), ('Yes', 'No'))
ax[0, 1].bar(deposit_yes['month'].unique(),height = deposit_yes['month'].value_counts(),color='#000000')
ax[0, 1].bar(deposit_no['month'].unique(),height = deposit_no['month'].value_counts(),bottom = deposit_yes['month'].value_counts(),color = '#DC4405') 
ax[0, 1].title.set_text('Month')
ax[1, 0].bar(deposit_yes['job'].unique(),height = deposit_yes['job'].value_counts(),color='#000000')
ax[1, 0].bar(deposit_yes['job'].unique(),height = deposit_no['job'].value_counts()[deposit_yes['job'].value_counts().index],bottom = deposit_yes['job'].value_counts(),color = '#DC4405') 
ax[1, 0].title.set_text('Type of Job')
ax[1, 0].tick_params(axis='x',rotation=90)
ax[1, 1].bar(deposit_yes['education'].unique(),height = deposit_yes['education'].value_counts(),color='#000000') #row=0, col=1
ax[1, 1].bar(deposit_yes['education'].unique(),height = deposit_no['education'].value_counts()[deposit_yes['education'].value_counts().index],bottom = deposit_yes['education'].value_counts(),color = '#DC4405') 
ax[1, 1].title.set_text('Education')
ax[1, 1].tick_params(axis='x',rotation=90)
#ax[0, 1].xticks(rotation=90)
plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(15,10))

b1 = ax[0, 0].bar(deposit_yes['marital'].unique(),height = deposit_yes['marital'].value_counts(),color='#000000')
b2 = ax[0, 0].bar(deposit_yes['marital'].unique(),height = deposit_no['marital'].value_counts()[deposit_yes['marital'].value_counts().index],bottom = deposit_yes['marital'].value_counts(),color = '#DC4405') 
ax[0, 0].title.set_text('Marital Status')
#ax[0, 0].legend((b1[0], b2[0]), ('Yes', 'No'))
ax[0, 1].bar(deposit_yes['housing'].unique(),height = deposit_yes['housing'].value_counts(),color='#000000')
ax[0, 1].bar(deposit_yes['housing'].unique(),height = deposit_no['housing'].value_counts()[deposit_yes['housing'].value_counts().index],bottom = deposit_yes['housing'].value_counts(),color = '#DC4405') 
ax[0, 1].title.set_text('Has housing loan')
ax[0, 2].bar(deposit_yes['loan'].unique(),height = deposit_yes['loan'].value_counts(),color='#000000')
ax[0, 2].bar(deposit_yes['loan'].unique(),height = deposit_no['loan'].value_counts()[deposit_yes['loan'].value_counts().index],bottom = deposit_yes['loan'].value_counts(),color = '#DC4405') 
ax[0, 2].title.set_text('Has personal loan')
ax[1, 0].bar(deposit_yes['contact'].unique(),height = deposit_yes['contact'].value_counts(),color='#000000')
ax[1, 0].bar(deposit_yes['contact'].unique(),height = deposit_no['contact'].value_counts()[deposit_yes['contact'].value_counts().index],bottom = deposit_yes['contact'].value_counts(),color = '#DC4405') 
ax[1, 0].title.set_text('Type of Contact')
ax[1, 1].bar(deposit_yes['default'].unique(),height = deposit_yes['default'].value_counts(),color='#000000')
ax[1, 1].bar(deposit_yes['default'].unique(),height = deposit_no['default'].value_counts()[deposit_yes['default'].value_counts().index],bottom = deposit_yes['default'].value_counts(),color = '#DC4405') 
ax[1, 1].title.set_text('Has credit in default')
ax[1, 2].bar(deposit_yes['poutcome'].unique(),height = deposit_yes['poutcome'].value_counts(),color='#000000')
ax[1, 2].bar(deposit_yes['poutcome'].unique(),height = deposit_no['poutcome'].value_counts()[deposit_yes['poutcome'].value_counts().index],bottom = deposit_yes['poutcome'].value_counts(),color = '#DC4405') 
ax[1, 2].title.set_text('Outcome of the previous marketing campaign')
plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(12,10))

ax[0, 0].fitting(deposit_no['age'],color = '#DC4405',alpha=0.7,bins=20, edgecolor='white') 
ax[0, 0].fitting(deposit_yes['age'],color='#000000',alpha=0.5,bins=20, edgecolor='white')
ax[0, 0].title.set_text('Age')
ax[0, 1].fitting(deposit_no['duration'],color = '#DC4405',alpha=0.7, edgecolor='white') 
ax[0, 1].fitting(deposit_yes['duration'],color='#000000',alpha=0.5, edgecolor='white')
ax[0, 1].title.set_text('Contact duration')
ax[1, 0].fitting(deposit_no['campaign'],color = '#DC4405',alpha=0.7, edgecolor='white') 
ax[1, 0].fitting(deposit_yes['campaign'],color='#000000',alpha=0.5, edgecolor='white')
ax[1, 0].title.set_text('Number of contacts performed')
ax[1, 1].fitting(deposit_no[deposit_no['pdays'] != 999]['pdays'],color = '#DC4405',alpha=0.7, edgecolor='white') 
ax[1, 1].fitting(deposit_yes[deposit_yes['pdays'] != 999]['pdays'],color='#000000',alpha=0.5, edgecolor='white')
ax[1, 1].title.set_text('Previous contact days')
plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")
plt.show()

predictors = df.iloc[:,0:20]
print(df.columns)

#Label encoding

X = df.drop(columns=["deposit"],axis=1)
y = df[["deposit"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train_norm, X_test_norm = preprocessing.normalize(X_train), preprocessing.normalize(X_test)

perceptron_model = lm.Perceptron()
perceptron_model.fit(X_train,y_train)
y_pred = perceptron_model.predict(X_test)
print("Accuracy: ",(accuracy_score(y_test, y_pred)))

X_train_norm, X_test_norm = preprocessing.normalize(X_train), preprocessing.normalize(X_test)
batch_size = 10
num_classes = 10
epochs = 50
stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

model = Sequential([
    Dense(256, activation='relu', input_dim= 16),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()



model.compile(optimizer=RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

fitting = model.fit(X_train_norm,
                  y_train,
                  epochs=50,
                  batch_size=10,
                  validation_split=0.2,
                  callbacks=[])
# summarize history for accuracy
plt.plot(fitting.history['accuracy'])
plt.plot(fitting.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(fitting.history['loss'])
plt.plot(fitting.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



predicted_labels = (model.predict(X_test_norm))
predictions = [float(round(x[0])) for x in predicted_labels]
FPR, TPR, cutoffs = metrics.roc_curve(y_test,predictions,pos_label=1) 
bp_x = FPR
bp_y = TPR

plt.plot(bp_x, bp_y, linewidth=3,
         color="blue", label=r"Curve")
plt.xlabel(r"False Positive")
plt.ylabel(r"True Positive")
plt.title(r"Title here (remove for papers)")
plt.legend(loc="lower left")
plt.show()
plt.show()

score = model.evaluate(X_test_norm, y_test, verbose=0)
print('Test accuracy:', score[1])


def conf_matrix(y_test, y_pred, model):
    cm = confusion_matrix(y_test, y_pred, labels=range(0,2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(0,2))
    fig, ax = plt.subplots(figsize=(15,12))
    disp.plot(ax=ax)
    plt.show()
conf_matrix(y_test, predictions, model)

#Unnormalized data
fitting1 = model.fit(X_train,
                 y_train,
                 epochs=50,
                 batch_size=10,
                 validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])


