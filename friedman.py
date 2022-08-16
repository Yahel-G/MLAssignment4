import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp

khan = pd.read_csv(r'results\08-16-2022_21-12-31_data=khanauc.csv').iloc[:,[5,11]]
leukemia = pd.read_csv(r'results\08-16-2022_21-13-08_data=leukemiaauc.csv').iloc[:,[5,11]]
cll = pd.read_csv(r'results\08-16-2022_21-10-16_data=cllauc.csv').iloc[:,[5,11]]
bladderbatch = pd.read_csv(r'results\08-16-2022_21-10-49_data=bladderbatchauc.csv').iloc[:,[5,11]]
sorlie = pd.read_csv(r'results\08-16-2022_21-11-05_data=sorlieauc.csv').iloc[:,[5,11]]
subramanian = pd.read_csv(r'results\08-16-2022_21-11-37_data=subramanianauc.csv').iloc[:,[5,11]]

f_classif_auc =  pd.concat([khan['Measure Value'].loc[khan['Filtering Algorithm'] == 'f_classif_wrapper'],\
           leukemia['Measure Value'].loc[leukemia['Filtering Algorithm'] == 'f_classif_wrapper'],\
               cll['Measure Value'].loc[cll['Filtering Algorithm'] == 'f_classif_wrapper'],\
                   bladderbatch['Measure Value'].loc[bladderbatch['Filtering Algorithm'] == 'f_classif_wrapper'],\
                       sorlie['Measure Value'].loc[sorlie['Filtering Algorithm'] == 'f_classif_wrapper'],\
                           subramanian['Measure Value'].loc[subramanian['Filtering Algorithm'] == 'f_classif_wrapper']])
relaxed_lasso_auc =  pd.concat([khan['Measure Value'].loc[khan['Filtering Algorithm'] == 'relaxed_lasso_wrapper'],\
           leukemia['Measure Value'].loc[leukemia['Filtering Algorithm'] == 'relaxed_lasso_wrapper'],\
               cll['Measure Value'].loc[cll['Filtering Algorithm'] == 'relaxed_lasso_wrapper'],\
                   bladderbatch['Measure Value'].loc[bladderbatch['Filtering Algorithm'] == 'relaxed_lasso_wrapper'],\
                       sorlie['Measure Value'].loc[sorlie['Filtering Algorithm'] == 'relaxed_lasso_wrapper'],\
                           subramanian['Measure Value'].loc[subramanian['Filtering Algorithm'] == 'relaxed_lasso_wrapper']])
RFE_auc =  pd.concat([khan['Measure Value'].loc[khan['Filtering Algorithm'] == 'RFE_wrapper'],\
           leukemia['Measure Value'].loc[leukemia['Filtering Algorithm'] == 'RFE_wrapper'],\
               cll['Measure Value'].loc[cll['Filtering Algorithm'] == 'RFE_wrapper'],\
                   bladderbatch['Measure Value'].loc[bladderbatch['Filtering Algorithm'] == 'RFE_wrapper'],\
                       sorlie['Measure Value'].loc[sorlie['Filtering Algorithm'] == 'RFE_wrapper'],\
                           subramanian['Measure Value'].loc[subramanian['Filtering Algorithm'] == 'RFE_wrapper']])
relieff_auc =  pd.concat([khan['Measure Value'].loc[khan['Filtering Algorithm'] == 'relieff_wrapper'],\
           leukemia['Measure Value'].loc[leukemia['Filtering Algorithm'] == 'relieff_wrapper'],\
               cll['Measure Value'].loc[cll['Filtering Algorithm'] == 'relieff_wrapper'],\
                   bladderbatch['Measure Value'].loc[bladderbatch['Filtering Algorithm'] == 'relieff_wrapper'],\
                       sorlie['Measure Value'].loc[sorlie['Filtering Algorithm'] == 'relieff_wrapper'],\
                           subramanian['Measure Value'].loc[subramanian['Filtering Algorithm'] == 'relieff_wrapper']])

print(stats.friedmanchisquare(f_classif_auc, relaxed_lasso_auc, RFE_auc, relieff_auc))

data = np.array([f_classif_auc, relaxed_lasso_auc, RFE_auc, relieff_auc])
print(sp.posthoc_nemenyi_friedman(data.T))

