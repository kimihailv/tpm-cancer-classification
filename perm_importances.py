import numpy as np
import pandas as pd
import sklearn.ensemble
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from time import time
from sklearn.externals import joblib
from functools import partial

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

def perm_importances(args):
    clf, cls = args
    row = {'class': cls}
    y = y_test.copy()
    x = X_test.copy()
    y[y != cls] = -1
    y[y == cls] = 1
    y[y == -1] = 0

    pred = clf.predict(x)
    baseline = accuracy_score(y, pred)
    print('Baseline for {} class: {}'.format(cls, baseline))
    for col in range(17812):
        save = x[:,col].copy()
        x[:,col] = np.random.permutation(x[:,col])
        pred = clf.predict(x)
        acc = accuracy_score(y, pred)
        row[col] = baseline - acc
        x[:,col] = save
    return row

model = joblib.load('model.pkl')
print('Model is loaded')
st = time()
data = joblib.Parallel(n_jobs=10, require='sharedmem')(map(joblib.delayed(perm_importances), zip(model.estimators_, model.classes_)))
feat = pd.DataFrame(data)
feat.to_csv('features_meta_raw.csv')
print('Selection finished: {}'.format(time() - st))
