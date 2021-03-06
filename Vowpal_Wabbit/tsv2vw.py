import json
import pandas as pd

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
            X_all['body'][row] = X_all['boilerplate'][row]['body'].encode( 'ascii', 'ignore' ).replace('\n',' ').replace('\r',' ').replace(':',' ')

o = open('train.vw', 'w' )

for row in X.index:
	if y[row] == 0:
		y[row] = -1
	o.write("%s | %s\n" % (y[row],X_all['body'][row]))

o = open('test.vw', 'w' )

for row in X_test.index:
	o.write("%s | %s\n" % (1,X_all['body'][row]))

