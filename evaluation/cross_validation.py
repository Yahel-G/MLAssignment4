from sklearn.model_selection import cross_val_score

def cv_score(X, y, clf):
    k, cv_method  = find_k_cv(X)
    scores = cross_val_score(
        clf, X, y, cv=k, scoring='roc_auc_score_micro')

    return scores, cv_method, k 


def find_k_cv(X):
    n_samples = len(X)
    if n_samples < 50:
        cv_method =  'Leave-pair-out'
        k = n_samples // 2
        k = 2 # had to go with k = 2 otherwise I'd get a ton of errors
    elif n_samples < 100:
        cv_method =  'Leave-one-out'
        k = n_samples
        k = 2 # had to go with k = 2 otherwise I'd get a ton of errors
    elif n_samples < 1000:
        cv_method =  '10FoldsCV'
        k = 10
    else:
        cv_method =  '5FoldsCV'
        k = 5
    return k, cv_method 
