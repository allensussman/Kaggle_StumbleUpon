""" Naive Bayes Starter Code - StumbleUpon Kaggle Competition
bensolucky@gmail.com 
Kaggle: BS Man
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import load_svmlight_file

bodies = load_svmlight_file('../text2libsvm/train.txt')

print 'hello'

# Fit a model and predict
model = BernoulliNB()
model.fit(bodies, y)

# Added following three lines
train_preds = model.predict_proba(bodies)[:,1]
train_pred_df = pd.DataFrame(train_preds, index=X.index, columns=['label'])
train_pred_df.to_csv('train_p.csv')


preds = model.predict_proba(bodies_test)[:,1]
pred_df = pd.DataFrame(preds, index=X_test.index, columns=['label'])
pred_df.to_csv('test.csv')
