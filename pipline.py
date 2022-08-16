import shutil
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

from dataset.load_dataset import Dataset
from dataset.pre_process import check_label
from evaluation.cross_validation import find_k_cv
from evaluation.measures import evaluate
from feature_selection.swichclf import ClfSwitcher
from feature_selection.wrappers import (RFE_wrapper, 
                                        f_classif_wrapper, mrmr_wrapper,
                                        relaxed_lasso_wrapper, relieff_wrapper)

warnings.simplefilter("ignore")

def select_k_best(X,y,k=1000):
    f_scores = f_classif_wrapper(np.array(df['X']), np.array(df['y']))
    max_features_idx = np.argsort(f_scores[::-1])[:k]
    new_df = {'X':df['X'].iloc[:,max_features_idx], 'y':df['y']}
    return new_df
    
# Load dataset
dataset_names = ['all','bladderbatch','breastcancervdx','cll','curatedovariandata',
                'bp','cbh','cs','css','pdx',
                'khan', 'sorlie', 'subramanian', 'sun', 'yeoh',
                'gli-85','leukemia','smk-can-187','tox-171','usps'] # bladderbatch cll sorlie subramanian khan leukemia


name = 'leukemia'

# for i in range(len(dataset_names)):
#     dat = Dataset(dataset_names[i]).load()
#     n_feats = dat['X'].shape[1]
#     n_sam = dat['X'].shape[0]
#     print(dataset_names[i] + " " + str(n_feats) + " " + str(n_sam))
    


data = Dataset(name)
df = data.load()
df = check_label(df)

df = select_k_best(df['X'], df['y'])

X_train, X_test, y_train, y_test  = train_test_split(df['X'], df['y'], test_size=0.2, random_state=0, stratify=df['y'])
MULTICLASS = [True if len(np.unique(y_train)) > 2 else False]
def pr_auc_score(y_true, y_score):
    prauc = average_precision_score(y_true, y_score, average='micro')
    return prauc

if MULTICLASS:
    def auc_score(y_true, y_score):
        auc = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
        return auc
    
else:
    
    def auc_score(y_true, y_score):
        auc = roc_auc_score(y_true, y_score)
        return auc

# pr_auc = make_scorer(score_func=pr_auc_score,  needs_proba=True)
auc = make_scorer(score_func=auc_score,greater_is_better=True, needs_proba=True)

SCORING = {'accuracy':'accuracy',
           'matthews_corrcoef':'matthews_corrcoef',
           'pr_auc':'precision_micro',
           'roc_auc':auc,
           }


main_pipe = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance', VarianceThreshold()),
        ('power', PowerTransformer()),
        ("reduce_dim", SelectKBest()),
        ('clf', ClfSwitcher()),
        ],
        memory=r".\cache",
        verbose=True,
)

# HYPER PARAMETERS
K_FEATURES_OPTIONS = [1,2,3,4,5,10,20,50,100]
ESTIMATORS = [f_classif_wrapper, RFE_wrapper, relieff_wrapper, relaxed_lasso_wrapper] # f_classif_wrapper, RFE_wrapper, relieff_wrapper, relaxed_lasso_wrapper, mrmr_wrapper
K_CV, CV_METHOD = find_k_cv(X_train)

#pipeline parameters
parameters = [
        {
        "reduce_dim__score_func": ESTIMATORS,
        "reduce_dim__k": K_FEATURES_OPTIONS,
        'clf__estimator': [GaussianNB()],
        },
        {
        "reduce_dim__score_func": ESTIMATORS,
        "reduce_dim__k": K_FEATURES_OPTIONS,
        'clf__estimator': [SVC()],
        'clf__estimator__C': [0.1],
        'clf__estimator__probability': [True],
        'clf__estimator__kernel': ['linear'],
        'clf__estimator__random_state': [0],
        # 'clf__estimator__class_weight': ['balanced'],
        },
        {
        "reduce_dim__score_func": ESTIMATORS,
        "reduce_dim__k": K_FEATURES_OPTIONS,
        'clf__estimator': [RandomForestClassifier()],
        'clf__estimator__max_depth': [2],
        'clf__estimator__random_state': [0],
        # 'clf__estimator__class_weight': ['balanced'],
        },
        {
        "reduce_dim__score_func": ESTIMATORS,
        "reduce_dim__k": K_FEATURES_OPTIONS,
        'clf__estimator': [LogisticRegression()],
        'clf__estimator__C': [0.1],
        'clf__estimator__random_state': [0],
        # 'clf__estimator__class_weight': ['balanced'],
        },
        {
        "reduce_dim__score_func": ESTIMATORS,
        "reduce_dim__k": K_FEATURES_OPTIONS,
        'clf__estimator': [KNeighborsClassifier()],
        },
    ]


gscv = GridSearchCV(main_pipe, parameters, cv=K_CV, n_jobs=-1, return_train_score=True, verbose=10, scoring=SCORING, refit='accuracy')#, error_score='raise')
gscv.fit(X_train, np.squeeze(y_train))

time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
save_path = Path('results') / f'{time}_data={name}.npz'
Path('results').mkdir(exist_ok=True)
np.savez(save_path, gscv=gscv) # cv_results=cv_results,best_estimator=best_estimator,best_params=best_params,feature_names_in=feature_names_in)



# y_pred_test = best_estimator.predict(X_test)
# y_score_test = best_estimator.evaluate(X_test)
# acc, mcc, area, prauc = evaluate(y_test, y_pred_test, y_score_test, multiclass=MULTICLASS, display=True)


