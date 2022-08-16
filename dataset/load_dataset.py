from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat

# from dataset.pre_process import pre_process


class Dataset():

    def __init__(self, dataset):
        self.dataset_name = dataset
        self.main_path = Path('data')

    def load(self):
        if self.dataset_name.lower() == 'toy_example':
            data_path = self.main_path / self.dataset_name
            X_train = pd.read_csv(data_path / 'SPECTF.train', header=None)
            y_train = X_train.pop(0)
            X_test = pd.read_csv(data_path / 'SPECTF.test', header=None)
            y_test = X_test.pop(0)
            return X_train, y_train, X_test, y_test

        if self.dataset_name.lower() in ['all','bladderbatch','breastcancervdx','cll','curatedovariandata']:
            data_path = self.main_path / self.dataset_name
            data = pd.read_csv(str(data_path) + '.csv')
            self.df = {'y': data.iloc[0,1:],'X': data.iloc[1:,1:].T}
        
        if self.dataset_name.lower() in ['bp','cbh','cs','css','pdx']:
            data_path = self.main_path / self.dataset_name
            data = pd.read_csv(str(data_path) + '.csv', header=None)
            self.df = {'y': data.iloc[0,1:].astype('category'),'X': data.iloc[1:,1:].T}

        if self.dataset_name.lower() in ['khan', 'sorlie', 'subramanian', 'sun', 'yeoh']:
            data_path_input = self.main_path / str(self.dataset_name + '_inputs.csv')
            data_path_output = self.main_path / str(self.dataset_name + '_outputs.csv')
            data_X = pd.read_csv(data_path_input, header=None)
            data_y = pd.read_csv(data_path_output, header=None)
            self.df ={'y': pd.DataFrame(data_y),'X': pd.DataFrame(data_X)}
        
        if self.dataset_name.lower() in ['gli-85','leukemia','smk-can-187','tox-171','usps']:
            data_path = self.main_path / self.dataset_name
            data = loadmat(str(data_path) + '.mat')
            self.df = {'y': pd.DataFrame(np.squeeze(np.array(data['Y'])).astype(np.float64)), 'X': pd.DataFrame(data['X'])}

        return self.df
        
    def save_dataframe(self):
        self.savedir = self.main_path / 'processed' 
        self.savedir.mkdir(exist_ok=True)
        with open (self.savedir / (self.dataset_name + '_processed.pkl'), 'wb') as f:
            pickle.dump(self.df,f)
        f.close

    def load_dataframe(self):
        self.savedir = self.main_path / 'processed' 
        try:
            with open(self.savedir / (self.dataset_name + '_processed.pkl'), 'rb') as f:
                self.df = pickle.load(f)
            f.close
            return self.df
        except:
            raise ValueError('This dataset is not saved!')

