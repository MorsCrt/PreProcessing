import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime

df = pd.read_csv("Multi-Gender-done.csv",low_memory=False)

def valmeanfiller(dataframe):
    # missval_colname = Which columns have missing values and numerical
    missval_colname = np.array([col for col in dataframe.columns
                                if dataframe[col].isnull().any()])
    numeric_colname = dataframe.select_dtypes(
        include=['float64',
                 'float32',
                 'int64',
                 'int32']).columns.values
    # Intersection of the numeric_colname and missval_colname
    inter_colnames = np.intersect1d(missval_colname,
                                    numeric_colname)
    # Saving order name for after concat

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    df_misscol = dataframe[inter_colnames].values  # List to dataframe
    imputer = imputer.fit(df_misscol)
    df_misscol = imputer.transform(df_misscol)
    
     # Drop columns(have missing values) on the main dataframe
    df = dataframe.drop(inter_colnames, axis=1) 

    df_filled = pd.DataFrame(data=df_misscol,
                             index=range(len(df_misscol)),
                             columns=inter_colnames)
    
    # Concat mean filled table and dropped dataframe
    df = pd.concat([df, df_filled], axis=1)  

    return df


start_time = datetime.now()
df_meanfilled = valmeanfiller(df)
end_time = datetime.now()
df_meanfilled.to_csv("alldone.csv",index=False)
print('Duration: {}'.format(end_time - start_time))

