'Does model stacking on different model results using SVC'

import json
import numpy as np
import pandas as pd
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

SparseNN=pd.read_csv('../SparseNN/train_p.csv', sep=",", na_values=['?'], index_col=0)

Naive_Bayes=pd.read_csv('../Naive_Bayes/train_p.csv', sep=",", na_values=['?'], index_col=0)

Vowpal_Wabbit=pd.read_csv('../Vowpal_Wabbit/train_p.csv', sep=",", na_values=['?'], index_col=0)

Logit=pd.read_csv('../Logit/train_p.csv', sep=",", na_values=['?'], index_col=0)

train_preds=pd.DataFrame({'Naive_Bayes': Naive_Bayes['label'],'SparseNN':SparseNN['label'],'Vowpal_Wabbit':Vowpal_Wabbit['label'],'Logit':Logit['label']})

clf=svm.SVC(probability=True);

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

SparseNN=pd.read_csv('../SparseNN/p.csv', sep=",", na_values=['?'], index_col=0)

Naive_Bayes=pd.read_csv('../Naive_Bayes/test.csv', sep=",", na_values=['?'], index_col=0)

Vowpal_Wabbit=pd.read_csv('../Vowpal_Wabbit/test_p.csv', sep=",", na_values=['?'], index_col=0)

Logit=pd.read_csv('../Logit/test.csv', sep=",", na_values=['?'], index_col=0)

preds=pd.DataFrame({'Naive_Bayes': Naive_Bayes['label'],'SparseNN':SparseNN['label'],'Vowpal_Wabbit':Vowpal_Wabbit['label'],'Logit':Logit['label']})

meta_pred=clf.predict_proba(preds)[:,1]
meta_pred_df = pd.DataFrame(meta_pred, index=X.index, columns=['label'])
meta_pred_df.to_csv('meta_p_SVC.csv')