from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, LabelEncoder
import numpy as np
import pandas as pd

def check_label(df):
    # check labels type
    dtype = df['y'].dtypes
    try:
        dtype = dtype.item()
    except:
        pass

    if dtype == 'category':
        le = LabelEncoder()
        new_y = le.fit_transform(df['y'])
        df['y'] = pd.DataFrame(new_y)

    # replace nan label with new label
    df['y'] = df['y'].fillna(max(df['y'])+1)

    return df





