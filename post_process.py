
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm

PATH = Path(r"results")
datafiles = glob.glob(str(PATH / "*.csv"))
selected_features = pd.read_csv(PATH / '08-16-2022_21-13-11_selected_features.csv').drop(columns=['Unnamed: 0']) # '08-17-2022_00-00-47_selected_features.csv'
K = [1,2,3,4,5,10,20,50,100]
datasets = ['leukemia','khan','subramanian','sorlie','cll','bladderbatch'] # 'toy_example'
all_results = pd.DataFrame()
for name in tqdm(datasets):
    acc_file = [f for f in datafiles if name + 'ACC' in f]
    auc_file = [f for f in datafiles if name + 'auc' in f]
    MCC_file = [f for f in datafiles if name + 'MCC' in f]
    prauc_file = [f for f in datafiles if name + 'pr_auc' in f]

    acc = pd.read_csv(acc_file[0]).drop(columns=['Unnamed: 0','index'])
    auc = pd.read_csv(auc_file[0]).drop(columns=['Unnamed: 0','index'])
    mcc = pd.read_csv(MCC_file[0]).drop(columns=['Unnamed: 0','index'])
    prauc = pd.read_csv(prauc_file[0]).drop(columns=['Unnamed: 0','index'])

    #merged_df = pd.concat([acc,auc,mcc,prauc])
    merged_df = pd.concat([acc,auc,prauc])
    data_features = selected_features[selected_features['Dataset Name']==name]
    algs = merged_df['Filtering Algorithm'].unique()
    merged_df['List of Selected Features Names'] = pd.NaT
    merged_df['Selected Features scores'] = pd.NaT

    for alg in algs:
        data = data_features.loc[data_features['Filtering Algorithm']==alg]
        for k in K:
            try:
                feat_names = data['List of Selected Features Names'].loc[data['k']==str(k)] 
                feat_scores =  data['Selected Features scores'].loc[data['k']==str(k)] 

                merged_df.loc[(merged_df['Filtering Algorithm']==alg) & (merged_df['Number of features selected (K)']== k), 'List of Selected Features Names'] = feat_names.values[0]
                merged_df.loc[(merged_df['Filtering Algorithm']==alg) & (merged_df['Number of features selected (K)']== k), 'Selected Features scores'] = feat_scores.values[0]
            except:
                continue

    all_results = pd.concat([all_results,merged_df]).reset_index().drop(columns='index')

all_results.to_csv(PATH / 'all_results.csv')