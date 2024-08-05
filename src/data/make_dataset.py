import pandas as pd
import numpy as np


def load_data(data_path):
    df = pd.read_csv(data_path)
    
    print("Print Head",df.head())
    
    print("Data Shape",df.shape)
    
    print("Check data type",df.dtypes)
    
    return df



    
