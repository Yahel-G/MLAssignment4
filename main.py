import sys
print(sys.version)

import numpy as np
import pandas as pd
from feature_selection.relaxed_lasso import RelaxedLassoLarsCV
from dataset.load_dataset import Dataset
from evaluation.cross_validation import cv
from sklearn.linear_model._base import _preprocess_data
from sklearn.svm import SVR
from sklearn import preprocessing, feature_selection
from sklearn.feature_selection import SelectKBest
from scikit_feature import mrmr, reliefF, RFE, f_classif, feature_ranking, SelectFdr
from tqdm import tqdm
from subprocess import Popen, PIPE
from mlab import mlabwrap, mlabraw, releases
from mlab.releases import latest_release as matlab

# mlabraw.set_release('R2019')
# m = releases.MatlabVersions({'k':'R2014b'})
# instance = m.get_mlab_instance('R2014b')
# matlab.plot([1,2,3],'-o')
from numpy import *
matlab.main_USFS()
xx = arange(-2*pi, 2*pi, 0.2)
f = matlab.surf(subtract.outer(sin(xx),cos(xx)))
# mlabwrap.path(mlabwrap.path(), r'C:\School\Master\Courses\Computational Learning\Ass4\USFS-code-master')


## Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# clf = LogisticRegression(random_state=0).fit(X, y)
# clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y)
# clf = GaussianNB.fit(X, y)
# neigh = KNeighborsClassifier(n_neighbors=3).fit(X, y)
# svm = SVC(kernel= 'linear', random_state=1, C=0.1).fit(X, y)



# Relas <- relaxo(y2,ys,phi=1/3,max.steps = min(2*length(ys), 2 * ncol(y2)))

def run_R(file):
  # COMMAND WITH ARGUMENTS
  cmd = ["Rscript", "myR_script.R", file]

  p = Popen(cmd, cwd="/path/to/folder/of/my_script.R/", stdin=PIPE, stdout=PIPE, stderr=PIPE)     
  output, error = p.communicate()

  # PRINT R CONSOLE OUTPUT (ERROR OR NOT)
  if p.returncode == 0:            
      print('R OUTPUT:\n {0}'.format(output))            
  else:                
      print('R ERROR:\n {0}'.format(error))

dataset_names = ['all','bladderbatch','breastcancervdx','cll','curatedovariandata',
                'bp','cbh','cs','css','pdx',
                'khan', 'sorlie', 'subramanian', 'sun', 'yeoh',
                'gli-85','leukemia','smk-can-187','tox-171','usps'] 

# for name in tqdm(dataset_names):
#     data = Dataset(name)
#     data.load()
#     data.process()
#     data.save_dataframe()
#     # df = data.load_dataframe()
#     # relassoCV = RelaxedLassoLarsCV(cv=5,verbose=True,max_iter=min(2*len(df['y']), 2 * len(df['X'].columns)), n_jobs=-1).fit(df['X'], df['y'])


# part a: toy example
toy_example = Dataset('toy_example').load()
# mlab.addpath(r'C:\School\Master\Courses\Computational Learning\Ass4\USFS-code-master')
ugh = mlab.length(toy_example)

X_train, y_train, X_test, y_test = toy_example.load()
X_train, y_train = np.array(X_train), np.array(y_train)
n_features = X_train.shape[1]





relassoCV = RelaxedLassoLarsCV(cv=5,).fit(X_train, y_train) # cv=80
relassoCV_features = feature_selection.SelectKBest([RelaxedLassoLarsCV(cv=5).fit(X_train, y_train), 44])


mRMR_features = feature_selection.SelectKBest([mrmr(X_train, y_train, n_selected_features=n_features), 44])


relieff_features = feature_selection.SelectKBest([feature_ranking(reliefF(X_train, y_train)),44])



f_classif_Fdr_features = feature_selection.SelectKBest([SelectFdr(f_classif, alpha=0.1).fit(X_train, y_train), 44])
RFE_feat = RFE(SVR(kernel="linear"), n_features_to_select=1, step=1)
#RFE_features = feature_selection.SelectKBest([RFE_features.fit(X_train, y_train).ranking_, 44])
RFE_features = feature_selection.SelectKBest([RFE_feat.fit(X_train, y_train), 44])

# X_test, y_test, _, _, _ = _preprocess_data(X_test, y_test, fit_intercept=True, normalize=True, copy=True)
# print("R-squared: ", relassoCV.score(X_train, y_train))
y_pred = relassoCV.predict(X_train)
y_pred = np.rint(preprocessing.MinMaxScaler().fit_transform(y_pred.reshape(-1,1)))


# Best parameters
print("Best Alpha: ", relassoCV.alpha_)
print("Best Theta: ", relassoCV.theta_)


print(f"Relaxed lasso R-squared score on test set: {relassoCV.score(X_test, y_test)}")
print(f"""Predictors retained by relaxed lasso: {np.count_nonzero(relassoCV.coef_)}""")

print(f'dataset size: {len(X_train)}')








