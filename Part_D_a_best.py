import pandas as pd
import numpy as np
from pathlib import Path

khan = pd.read_csv(r'results\08-16-2022_21-12-31_data=khanauc.csv')
khan_best = khan.loc[khan['Measure Value'] == khan['Measure Value'].max()]
leukemia = pd.read_csv(r'results\08-16-2022_21-13-08_data=leukemiaauc.csv')
leukemia_best = leukemia.loc[leukemia['Measure Value'] == leukemia['Measure Value'].max()]
cll = pd.read_csv(r'results\08-16-2022_21-10-16_data=cllauc.csv')
cll_best = cll.loc[cll['Measure Value'] == cll['Measure Value'].max()]
bladderbatch = pd.read_csv(r'results\08-16-2022_21-10-49_data=bladderbatchauc.csv')
bladderbatch_best = bladderbatch.loc[bladderbatch['Measure Value'] == bladderbatch['Measure Value'].max()]
sorlie = pd.read_csv(r'results\08-16-2022_21-11-05_data=sorlieauc.csv')
sorlie_best = sorlie.loc[sorlie['Measure Value'] == sorlie['Measure Value'].max()]
subramanian = pd.read_csv(r'results\08-16-2022_21-11-37_data=subramanianauc.csv')
subramanian_best = subramanian.loc[subramanian['Measure Value'] == subramanian['Measure Value'].max()]
res = pd.concat([khan_best, leukemia_best, cll_best, bladderbatch_best, sorlie_best, subramanian_best])
res.to_csv(Path(r'results')/'Part_D_a_best.csv') 
hi = 1