import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation

# Set a seed for consistant results
###############################################################################
# Load Data into pandas and Preprocess Features
###############################################################################
X = pd.read_csv('data/train.tsv', sep="\t", na_values=['?'], index_col=1)
X_test = pd.read_csv('data/test.tsv', sep="\t", na_values=['?'], index_col=1)
y = X['label']
X = X.drop(['label'], axis=1)
# Combine test and train while we do our preprocessing
X_all = pd.concat([X_test, X])

X_all['boilerplate'] = X_all['boilerplate'].apply(json.loads)
# Initialize the data as a unicode string
X_all['body'] = u'empty'
# Run through the dataframe row-by-row, building the new columns with info
# extracted from the boilerplate
# NOTE: Works fine on this small dataset, but probably not the most efficient
# way to do this, if you have any tips please let me know.
for row in X_all.index:
    # First check that the body exists
    if 'body' in X_all['boilerplate'][row].keys():
	# If the field exists but the value is missing, replace with a custom
	# "empty" flag.  Otherwise the CountVectorizer below will breaks.
	if pd.isnull(X_all['boilerplate'][row]['body']):
	    # the other values in 'body' are unicode strings 
	    X_all['body'][row] = u'empty'
	else:
            X_all['body'][row] = X_all['boilerplate'][row]['body']
body_counter = CountVectorizer()
body_counts = body_counter.fit_transform(X_all['body'])
# Re-seperate the test and training rows
bodies = body_counts[len(X_test.index):]
bodies_test = body_counts[:len(X_test.index)]

# Fit a model and predict
model = LogisticRegression()
model.fit(bodies, y)

print np.mean(cross_validation.cross_val_score(model, bodies, y, cv=10, scoring='roc_auc')) # 0.802135658218

# Added following three lines
train_preds = model.predict_proba(bodies)[:,1]
train_pred_df = pd.DataFrame(train_preds, index=X.index, columns=['label'])
train_pred_df.to_csv('train_p.csv')


preds = model.predict_proba(bodies_test)[:,1]
pred_df = pd.DataFrame(preds, index=X_test.index, columns=['label'])
pred_df.to_csv('test.csv')
