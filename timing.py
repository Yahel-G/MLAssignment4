

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_score(y_true, y_score):
    auc = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
    return auc


PATH = Path("results")

results_files = glob.glob(str(PATH / "*.npz"))
all_timings = pd.DataFrame(columns={'Dataset Name', "Fit Time (Mean)", "Fit Time (STD)",
                               "Score Time (Mean)", "Score Time (STD)",
                               "Estimator", "Number of features selected (K)",
                                "Score Function"})
for file in results_files:
    csv_filename = file.split('.')[0] + '.csv'
    results = np.load(file, allow_pickle=True)
    results = results['gscv'].data.obj.item(0)
    data_name = file.split('=')[-1][:-4]
    cv_results = pd.DataFrame(results.cv_results_)
    cv_results = cv_results.rename(columns={"mean_fit_time": "Fit Time (Mean)", "std_fit_time": "Fit Time (STD)",
                               "mean_score_time": "Score Time (Mean)", 'std_score_time': "Score Time (STD)",
                               "param_clf__estimator": "Estimator", "param_reduce_dim__k": "Number of features selected (K)",
                               "param_reduce_dim__score_func": "Score Function"})
    cv_results = cv_results.iloc[:,:7]
    cv_results['Dataset Name'] = data_name
    cv_results.iloc[:,-2] = cv_results.iloc[:,-2].map(lambda a: str(a).split(' ')[1])
    cv_results.loc[:,"Estimator"] = cv_results.loc[:,"Estimator"].map(lambda a: str(a).split('(')[0])

    all_timings = pd.concat([all_timings, pd.DataFrame(cv_results)])

all_timings = all_timings.drop(columns=['Score Time (STD)', 'Fit Time (STD)'])
all_timings.to_csv(PATH / "timings.csv")
