import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime

df = pd.read_csv("eksikveriler.csv")


def valmeanfiller(dataframe):
    global df
    # missval_colname = Which columns have missing values and numerical
    missval_colname = np.array([col for col in df.columns
                                if df[col].isnull().any()])
    numeric_colname = df.select_dtypes(
        include=['float64', 'int64']).columns.values
    # Intersection of the numeric_colname and missval_colname
    inter_colnames = np.intersect1d(missval_colname, numeric_colname)
    # Saving order name for after concat
    all_col_name = df.columns.values

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    df_misscol = dataframe[inter_colnames].values  # List to dataframe
    imputer = imputer.fit(df_misscol)
    df_misscol = imputer.transform(df_misscol)

    df_filled = pd.DataFrame(data=df_misscol,
                             index=range(len(df_misscol)),
                             columns=inter_colnames)

    df = df.drop(inter_colnames, axis=1)  # Drop columns(have missing values) on the main dataframe
    df = pd.concat([df, df_filled], axis=1)  # Concat mean filled table and dropped dataframe
    df = df[all_col_name]

    return df


start_time = datetime.now()
x = valmeanfiller(df)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
