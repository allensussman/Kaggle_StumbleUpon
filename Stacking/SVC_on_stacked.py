""" Naive Bayes Starter Code - StumbleUpon Kaggle Competition
bensolucky@gmail.com 
Kaggle: BS Man
"""

import json
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Set a seed for consistant results
###############################################################################
# Load Data into pandas and Preprocess Features
###############################################################################
# Train model
X = pd.read_csv('data/train.tsv', sep="\t", na_values=['?'], index_col=1)
y = X['label']

Zygmuntz=pd.read_csv('train_p_Zygmuntz+.csv', sep=",", na_values=['?'], index_col=0)

BS_Man=pd.read_csv('train_p_BS_Man+.csv', sep=",", na_values=['?'], index_col=0)

train_preds=pd.DataFrame({'BS_Man': BS_Man['label'],'Zygmuntz':Zygmuntz['label']})

clf=svm.SVC(probability=True);
clf.fit(train_preds,y)
meta_train_pred=clf.predict_proba(train_preds)[:,1]
meta_train_pred_df = pd.DataFrame(meta_train_pred, index=X.index, columns=['label'])
meta_train_pred_df.to_csv('meta_train_p.csv')

# Make prediction
X = pd.read_csv('data/test.tsv', sep="\t", na_values=['?'], index_col=1)
# X_test = pd.read_csv('data/test.tsv', sep="\t", na_values=['?'], index_col=1)

Zygmuntz=pd.read_csv('p_Zygmuntz+.csv', sep=",", na_values=['?'], index_col=0)

BS_Man=pd.read_csv('p_BS_Man+.csv', sep=",", na_values=['?'], index_col=0)

preds=pd.DataFrame({'BS_Man': BS_Man['label'],'Zygmuntz':Zygmuntz['label']})

meta_pred=clf.predict_proba(preds)[:,1]
meta_pred_df = pd.DataFrame(meta_pred, index=X.index, columns=['label'])
meta_pred_df.to_csv('meta_p.csv')