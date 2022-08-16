import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

from dataset.load_dataset import Dataset
from dataset.pre_process import check_label
from feature_selection.wrappers import (RFE_wrapper, f_classif_wrapper,
                                        mrmr_wrapper, relaxed_lasso_wrapper,
                                        relieff_wrapper)

SEED = 0
warnings.simplefilter("ignore")
np.random.seed(SEED)
def select_k_best(X,y,k=1000):
    f_scores = f_classif_wrapper(np.array(df['X']), np.array(df['y']))
    max_features_idx = np.argsort(f_scores[::-1])[:k]
    new_df = {'X':df['X'].iloc[:,max_features_idx], 'y':df['y']}
    return new_df
    
# Load dataset
dataset_names = ['toy_example'] ##### 'usps',################ ['cll', 'bladderbatch', 'sorlie', 'usps', 'subramanian', 'khan', 'leukemia'] 
time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
selected_features = pd.DataFrame(columns=['Dataset Name','Filtering Algorithm', 'k','List of Selected Features Names','Selected Features scores'])
for name in tqdm(dataset_names):
    data = Dataset(name)
    if name == 'toy_example':
        X_train, y_train, X_test_final, y_test_final = data.load()
        idx = np.random.permutation(X_train.index)
        X_train, X_test, y_train, y_test  = train_test_split(X_train.reindex(idx),y_train.reindex(idx), test_size=0.2, random_state=SEED)
        feature_names = [str(i) for i in range(len(X_train.T))]
        
    else:
        df = data.load()
        df = check_label(df)
        df = select_k_best(df['X'], df['y'])

        X_train, X_test, y_train, y_test  = train_test_split(df['X'], df['y'], test_size=0.2, random_state=SEED)
        feature_names = df['X'].columns
    
     
    

    X_train = SimpleImputer(strategy='mean').fit_transform(X_train,y_train)
    X_train  = VarianceThreshold().fit_transform(X_train,y_train)
    X_train = PowerTransformer().fit_transform(X_train,y_train)

    #K_FEATURES_OPTIONS = [1,2,3,4,5,10,20,50,100]
    K_FEATURES_OPTIONS = [1,2,3,4,5,10,15,20,30,50,100]

    ESTIMATORS = [relaxed_lasso_wrapper, RFE_wrapper, f_classif_wrapper, relieff_wrapper, mrmr_wrapper] # relaxed_lasso_wrapper, RFE_wrapper, f_classif_wrapper, relieff_wrapper, mrmr_wrapper

    
    save_path = Path(r'results\toy') / f'{time}_selected_features.csv' ### remove \relaxo
    Path(r'results').mkdir(exist_ok=True)

    for estimator in ESTIMATORS:

        est_name = str(estimator).split('function')[1].split('at')[0].strip()
        try:
            best_feat_scores = estimator(X_train, y_train)
            best_feat_idx = np.argsort(best_feat_scores)
            best_feat_names = feature_names[best_feat_idx]
        except:
            best_k_feat_idx = 'NAN'
            best_k_feat_names = 'NAN'
            res = pd.DataFrame({'Dataset Name': name, 'Filtering Algorithm': est_name, 'k': 'NAN', 'List of Selected Features Names': [best_k_feat_names],'Selected Features scores': [best_feat_scores]})
            selected_features = pd.concat([selected_features, res])
            selected_features.to_csv(save_path)
            continue

        for k in K_FEATURES_OPTIONS:
            best_k_feat_idx = best_feat_idx[:k]
            best_k_feat_names = feature_names[best_k_feat_idx]
            res = pd.DataFrame({'Dataset Name': name, 'Filtering Algorithm': est_name, 'k': k,'List of Selected Features Names': [best_k_feat_names.values],'Selected Features scores': [best_feat_scores]})
            selected_features = pd.concat([selected_features, res])
            selected_features.to_csv(save_path)



