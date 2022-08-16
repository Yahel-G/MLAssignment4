import pandas as pd
import numpy as np
## R stuff
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
#Must be activated
pandas2ri.activate()
from rpy2.robjects import numpy2ri
#Must be activated
numpy2ri.activate()
r.library("relaxo")


# r_y_t = pd.DataFrame(np.array(r_y),columns=r_X.names[1])

def relaxo(X, y):
    y_scaled = (y - np.mean(y))/np.std(y)
    X_scaled = (X - np.mean(X))/np.std(X)
    relaxo_object = robjects.r('relaxo')

    relas = relaxo_object(X_scaled, Y=y_scaled.values, phi=1/3, max_steps=min(2*len(y), 2*X.shape[1]))
    # r_X = relas.rx2('X')
    # r_y = relas.rx2('Y')
    r_phi = relas.rx2('phi')
    r_lambda = relas.rx2('lambda')
    r_beta = relas.rx2('beta')

    
    
    relen = np.zeros(4*len(y))
    n_features = X.shape[1]
    for i in range(len(r_lambda)):
        relen[i] = len(np.where(r_beta[i] != 0)[0])
        if relen[i] == n_features - 1 and r_phi[i] == 1/3:
            relaxo_features = np.argpartition(r_beta[i],n_features - 1)[:n_features - 1]
    
    # If we wanted all the options for feature selections:
    refram = [None] * n_features  # np.zeros((X.shape[1],))
    for i in range(len(r_lambda)):
        relen[i] = len(np.where(r_beta[i] != 0)[0])
        for j in range(1,n_features):
            if relen[i] == j and r_phi[i] == 1/3:
                # refram.append(np.where(r_beta[i] != 0)) 
                refram[j] = (np.argpartition(r_beta[i],j)[:j])
                #refram[j] = r_beta[i,np.where(r_beta[i]!=0)]
    
    # return relaxo_features
    z = np.zeros([min([n_features,1000])])
    z[np.array(refram[int(max(relen)-1)])] = 1
    return z.astype(int) # np.concatenate([np.array(refram[int(max(relen))]), np.squeeze(np.array((np.where(r_beta[int(max(relen))]==0))))])
