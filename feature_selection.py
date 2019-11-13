import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def log_importances(clf):
    rows = []
    for cls, est in zip(clf.classes_, clf.estimators_):
        row = {'class': cls}
        row.update(dict(enumerate(est.feature_importances_)))
        rows.append(row)
    return rows

def parse(x):
    return np.fromstring(x, sep=' ')

reader = pd.read_csv('full_ds.csv', chunksize=3000)
chunks = []

for chunk in reader:
    chunks.append(chunk)

ds = pd.concat(chunks).sample(frac=1)
del chunks
X = list(ds['features'])
X = map(parse, X)
X = np.array(list(X))
y = np.array(ds['class'].tolist())

clf = RandomForestClassifier(n_jobs=5, n_estimators=150)
onevsrest = OneVsRestClassifier(clf, n_jobs=10)

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)
data = []

i = 1
for train_index, test_index in skf.split(X, y):
    print('Start {} fold'.format(i))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    onevsrest.fit(X_train, y_train)
    pred = onevsrest.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print('Accuracy for {} fold: {}'.format(i, acc))
    rows = log_importances(onevsrest)
    data.extend(rows)
    print('Finish {} fold'.format(i))
    i += 1

df = pd.DataFrame(data)
df.to_csv('features_meta_raw1.csv')
