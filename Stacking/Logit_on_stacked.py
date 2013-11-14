""" Naive Bayes Starter Code - StumbleUpon Kaggle Competition
bensolucky@gmail.com 
Kaggle: BS Man
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics


# Set a seed for consistant results
###############################################################################
# Load Data into pandas and Preprocess Features
###############################################################################
# Train model
X = pd.read_csv('data/train.tsv', sep="\t", na_values=['?'], index_col=1)
y = X['label']

Zygmuntz=pd.read_csv('../Zygmuntz+/train_p.csv', sep=",", na_values=['?'], index_col=0)

BS_Man=pd.read_csv('../BS Man+/train_p.csv', sep=",", na_values=['?'], index_col=0)

Foxtrot=pd.read_csv('../Foxtrot+/train_p.csv', sep=",", na_values=['?'], index_col=0)

Logit=pd.read_csv('../Logit/train_p.csv', sep=",", na_values=['?'], index_col=0)

train_preds=pd.DataFrame({'BS_Man': BS_Man['label'],'Zygmuntz':Zygmuntz['label'],'Foxtrot':Foxtrot['label'],'Logit':Logit['label']})

clf=LogisticRegression();

print np.mean(cross_validation.cross_val_score(clf, train_preds, y, cv=3, scoring='roc_auc')) 

cv = cross_validation.KFold(len(train_preds), n_folds=20, indices=False)

#iterate through the training and test cross validation segments and
#run the classifier on each one, aggregating the results into a list
results = []
for traincv, testcv in cv:
    probas = clf.fit(train_preds[traincv], y[traincv]).predict_proba(train_preds[testcv])
    fpr, tpr, thresholds = metrics.roc_curve(y[testcv], [x[1] for x in probas], pos_label=1)
    results.append( metrics.auc(fpr,tpr) )
print results



clf.fit(train_preds,y)

meta_train_pred=clf.predict_proba(train_preds)[:,1]
meta_train_pred_df = pd.DataFrame(meta_train_pred, index=X.index, columns=['label'])
meta_train_pred_df.to_csv('meta_train_p.csv')



# Make prediction
X = pd.read_csv('data/test.tsv', sep="\t", na_values=['?'], index_col=1)
# X_test = pd.read_csv('data/test.tsv', sep="\t", na_values=['?'], index_col=1)

Zygmuntz=pd.read_csv('../Zygmuntz+/p.csv', sep=",", na_values=['?'], index_col=0)

BS_Man=pd.read_csv('../BS Man+/test.csv', sep=",", na_values=['?'], index_col=0)

Foxtrot=pd.read_csv('../Foxtrot+/test_p.csv', sep=",", na_values=['?'], index_col=0)

Logit=pd.read_csv('../Logit/test.csv', sep=",", na_values=['?'], index_col=0)

preds=pd.DataFrame({'BS_Man': BS_Man['label'],'Zygmuntz':Zygmuntz['label'],'Foxtrot':Foxtrot['label'],'Logit':Logit['label']})

meta_pred=clf.predict_proba(preds)[:,1]
meta_pred_df = pd.DataFrame(meta_pred, index=X.index, columns=['label'])
meta_pred_df.to_csv('meta_p_Logit.csv')