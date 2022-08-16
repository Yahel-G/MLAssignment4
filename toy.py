from sklearn.metrics import  roc_auc_score
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from dataset.load_dataset import Dataset
from evaluation.cross_validation import find_k_cv
def auc_score(y_true, y_score):
    auc = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
    return auc


PATH = Path(r'results') #### remove relaxo

results_files = glob.glob(str(PATH / "*.npz"))

for file in results_files:
    csv_filename = file.split('.')[0] #+ '.csv'
    results = np.load(file, allow_pickle=True)
    results = results['gscv'].data.obj.item(0)
    data_name = file.split('=')[-1][:-4]
    if data_name == 'toy_example':
        dataset, _, _, _ = Dataset(data_name).load()
    else:
        dataset = Dataset(data_name).load()
    n_samples = dataset.shape[0]
    n_orig_feat = dataset.shape[1]
    folds = results.cv
    classes = results.classes_
    f,cv_m = find_k_cv(dataset)
    cv_results = pd.DataFrame(results.cv_results_)
    # cv_results.columns
    cv_results['mean_fit_time']
    filtering_algorithm = cv_results.param_reduce_dim__score_func.values
    for p in range(len(filtering_algorithm)):
        filtering_algorithm[p] = str(filtering_algorithm[p]).split('function')[1].split('at')[0].strip()
    cv_method = pd.Series(cv_m).repeat(len(filtering_algorithm)).values
    
    n_orig_feat = pd.Series(n_orig_feat).repeat(len(filtering_algorithm)).values
    n_samples = pd.Series(n_samples).repeat(len(filtering_algorithm)).values
    data_name = pd.Series(data_name).repeat(len(filtering_algorithm)).values
    folds = pd.Series(folds).repeat(len(filtering_algorithm)).values
    pr_auc = pd.DataFrame({"Dataset Name": data_name, "Number of samples": n_samples, "Original Number of features": n_orig_feat, "Filtering Algorithm": filtering_algorithm, "Learning algorithm": cv_results.param_clf__estimator.values, "Number of features selected (K)": cv_results.param_reduce_dim__k.values, "CV Method": cv_method, "Fold": folds, "Measure Type": pd.Series("PR-AUC").repeat(len(filtering_algorithm)).values, "Measure Value": cv_results.mean_test_pr_auc.values}).reset_index()
    auc = pd.DataFrame({"Dataset Name": data_name, "Number of samples": n_samples, "Original Number of features": n_orig_feat, "Filtering Algorithm": filtering_algorithm, "Learning algorithm": cv_results.param_clf__estimator.values, "Number of features selected (K)": cv_results.param_reduce_dim__k.values, "CV Method": cv_method, "Fold": folds, "Measure Type": pd.Series("AUC").repeat(len(filtering_algorithm)).values, "Measure Value": cv_results.mean_test_roc_auc.values}).reset_index()
    ACC = pd.DataFrame({"Dataset Name": data_name, "Number of samples": n_samples, "Original Number of features": n_orig_feat, "Filtering Algorithm": filtering_algorithm, "Learning algorithm": cv_results.param_clf__estimator.values, "Number of features selected (K)": cv_results.param_reduce_dim__k.values, "CV Method": cv_method, "Fold": folds, "Measure Type": pd.Series("ACC").repeat(len(filtering_algorithm)).values, "Measure Value": cv_results.mean_test_accuracy.values}).reset_index()
    pr_auc.to_csv(csv_filename + 'pr_auc.csv')
    auc.to_csv(csv_filename + 'auc.csv')
    ACC.to_csv(csv_filename + 'ACC.csv')