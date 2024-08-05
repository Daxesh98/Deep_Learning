import pandas as pd
from sklearn.preprocessing import StandardScaler

# create dummy features
def dummy(df):
    
    Y= df.Attrition
    X= df.drop(columns = ['Attrition'])
    
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    
    return Y,X,X_scaled