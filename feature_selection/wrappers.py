import numpy as np
import sys
import os
path = os.path.join(os.getcwd(), 'scikit_feature')
sys.path.append(path)
from scikit_feature import mrmr, reliefF, RFE, f_classif, feature_ranking, SelectFdr
from sklearn.svm import SVR
from feature_selection.relaxo_r import relaxo
import mlab


def mrmr_wrapper(X, y):
    return mrmr(X, y, n_selected_features=X.shape[1])[0]

def f_classif_wrapper(X, y):
        # return np.argsort(SelectFdr(f_classif, alpha=0.1).fit(X, y).scores_)

        return SelectFdr(f_classif, alpha=0.1).fit(X, y).scores_
def relieff_wrapper(X, y):
    # return feature_ranking(reliefF(X, y))
    return reliefF(X, y)

def RFE_wrapper(X, y):
    rfe = RFE(SVR(kernel="linear"), n_features_to_select=1, step=1).fit(X, y).ranking_
    # print(rfe)
    return rfe

def relaxed_lasso_wrapper(X, y):
   return relaxo(X, y)

